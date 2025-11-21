import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import data_loader
# ==========================================
# 第一部分: 量子环境与电路定义
# ==========================================

# 定义量子设备
# n_qubits = 8 对应于我们 GCN 输出的特征维度
n_qubits = 8
n_layers = 2  # 变分电路的深度 (Layer number of VQC)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qlstm_circuit(inputs, weights):
    """
    QLSTM 的核心变分量子电路 (VQC)
    inputs: 输入特征 (归一化后的角度), 维度 (Batch, n_qubits)
    weights: 可训练的量子参数
    """
    # 1. 数据编码 (Angle Encoding) [cite: 11]
    # 将经典数据映射到量子态
    for i in range(n_qubits):
        qml.RY(inputs[:, i], wires=i) # 此时 inputs 包含 batch 维度，PennyLane 会自动广播
        
    # 2. 变分层 (Entanglement + Rotation) [cite: 15]
    # 用于学习特征变换
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    
    # 3. 测量 (Measurement)
    # 测量每个量子比特的 Pauli-Z 期望值，作为经典输出
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# ==========================================
# 第二部分: 结构门控 GCN (多模态融合核心)
# ==========================================

class StructureGatedGCN(nn.Module):
    def __init__(self, n_nodes=116, n_qubits=8):
        super(StructureGatedGCN, self).__init__()
        self.n_nodes = n_nodes
        self.n_qubits = n_qubits
        
        # 1. 结构门控网络 (sMRI -> Gate)
        # 学习 "灰质体积(GMV) -> 连接可靠性权重" 的非线性映射 [cite: 22]
        self.struct_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出 0~1，表示完整度
        )
        
        # 2. fMRI 特征变换 (Feature Mapping)
        # 将 fMRI 标量信号升维以匹配量子比特数
        self.fmri_linear = nn.Linear(1, n_qubits)
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, fmri, smri_gmv, adj_static):
        """
        参数:
        fmri: (Batch, Time, Nodes) - fMRI 时序数据
        smri_gmv: (Batch, Nodes) - sMRI 灰质体积数据
        adj_static: (Nodes, Nodes) - 静态平均功能连接矩阵
        """
        B, T, N = fmri.shape
        
        # === A. 结构门控 (Structure Gating) ===
        # 1. 计算每个节点的完整度权重
        # smri_gmv.unsqueeze(-1) -> (B, N, 1)
        node_integrity = self.struct_gate(smri_gmv.unsqueeze(-1)) 
        
        # 2. 生成动态掩码 (Dynamic Mask)
        # 两个脑区的连接权重 = ROI_i完整度 * ROI_j完整度
        # (B, N, 1) @ (B, 1, N) -> (B, N, N)
        struct_mask = node_integrity @ node_integrity.transpose(1, 2)
        
        # 3. 融合生成个性化动态图
        # 广播 adj_static: (1, N, N) * (B, N, N)
        adj_dynamic = adj_static.unsqueeze(0) * struct_mask
        
        # === B. 批处理动态图卷积 (Batch GCN) ===
        # 1. fMRI 特征升维
        # (B, T, N) -> (B, T, N, 1) -> (B, T, N, K)
        fmri_feat = self.fmri_linear(fmri.unsqueeze(-1))
        fmri_feat = F.relu(fmri_feat)
        
        # 2. 执行图卷积
        # 使用 einsum 一次性处理所有 Batch 和 Time 
        # 'bmn' (动态图), 'btnf' (fMRI特征) -> 'btmf' (新特征)
        # 物理含义: 聚合邻居信息，且受 sMRI 结构约束
        out = torch.einsum('bmn, btnf -> btmf', adj_dynamic, fmri_feat)
        out = F.relu(out)
        out = self.dropout(out)
        
        # 3. 图池化 (Readout)
        # 将 116 个节点的信息聚合为 8 维的全脑状态向量
        # (B, T, N, K) -> (B, T, K)
        out = torch.mean(out, dim=2)
        
        # 4. 归一化 (为量子编码做准备)
        # 使用 atan 将数值映射到 (-pi/2, pi/2) 附近，适合 Angle Encoding
        out = torch.atan(out)
        
        return out # 输出即为 QLSTM 的输入序列

# ==========================================
# 第三部分: 量子 LSTM 单元 (QLSTM Cell)
# ==========================================

class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2):
        super(QLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 定义量子层 (VQC)
        # 输入形状: n_qubits (即 hidden_size)
        # 输出形状: n_qubits
        # 我们用这个 VQC 来替换 LSTM 中处理输入的线性层 W_ih
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.vqc = qml.qnn.TorchLayer(qlstm_circuit, weight_shapes)
        
        # 经典线性层，用于将 VQC 的输出映射到 4 个门 (Input, Forget, Cell, Output)
        # 这里的输入是 VQC_out + Hidden_state
        self.cl_gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x):
        """
        x: 输入序列 (Batch, Time, Input_Size)
        """
        B, T, _ = x.size()
        
        # 初始化隐藏状态 h 和细胞状态 c
        h = torch.zeros(B, self.hidden_size).to(x.device)
        c = torch.zeros(B, self.hidden_size).to(x.device)
        
        # 存储所有时间步的输出
        outputs = []
        
        # 循环处理每个时间步
        for t in range(T):
            x_t = x[:, t, :] # 当前时刻输入 (B, input_size)
            
            # 1. 量子特征提取 (Quantum Transformation)
            # 将输入特征通过 VQC 进行非线性变换
            # 这是 QLSTM 的核心量子部分 [cite: 13]
            # x_t 已经是归一化过的角度数据
            vqc_out = self.vqc(x_t) # (B, hidden_size)
            
            # 2. 经典 LSTM 门控计算
            # 将量子变换后的输入与上一时刻的隐藏状态拼接
            combined = torch.cat((vqc_out, h), dim=1)
            
            # 计算门控 (i, f, g, o)
            gates = self.cl_gates(combined)
            i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)
            
            i = torch.sigmoid(i_gate)
            f = torch.sigmoid(f_gate)
            g = torch.tanh(g_gate)
            o = torch.sigmoid(o_gate)
            
            # 3. 更新细胞状态
            c = f * c + i * g
            
            # 4. 更新隐藏状态
            h = o * torch.tanh(c)
            
            outputs.append(h.unsqueeze(1))
            
        # 拼接所有时间步的输出
        return torch.cat(outputs, dim=1), (h, c)

# ==========================================
# 第四部分: Fused Q-MGRN 主模型
# ==========================================

class Fused_Q_MGRN(nn.Module):
    def __init__(self, n_nodes=116, n_time_steps=140, n_classes=2):
        super(Fused_Q_MGRN, self).__init__()
        
        # 超参数
        self.n_qubits = 8 
        self.pool_kernel = 4 # 时间池化窗口大小 (140 -> 35)
        
        # 模块 1: 结构门控 GCN (融合引擎)
        self.struct_gcn = StructureGatedGCN(n_nodes=n_nodes, n_qubits=self.n_qubits)
        
        # 模块 2: QLSTM (时序引擎)
        # 输入维度和隐藏维度都等于量子比特数
        self.qlstm = QLSTM(input_size=self.n_qubits, hidden_size=self.n_qubits, n_layers=2)
        
        # 模块 3: 分类器 (经典 Readout)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, fmri, smri, adj_static):
        """
        完整的前向传播路径
        """
        # 1. 结构门控与动态 GCN
        # 输入: fMRI(B, 140, 116), sMRI(B, 116), Adj(116, 116)
        # 输出: gcn_out (B, 140, 8) - 这里的 8 是 n_qubits
        gcn_out = self.struct_gcn(fmri, smri, adj_static)
        
        # 2. 时间池化 (Temporal Pooling) 
        # 优化策略: 将 140 步压缩，大幅减少量子电路执行次数
        # permute 为 (B, Feature, Time) 以使用 1D 池化
        gcn_out = gcn_out.permute(0, 2, 1) 
        gcn_out = F.avg_pool1d(gcn_out, kernel_size=self.pool_kernel)
        # 变回 (B, Time_short, Feature) -> (B, 35, 8)
        qlstm_in = gcn_out.permute(0, 2, 1)
        
        # 3. 量子 LSTM 处理
        # 输入: (B, 35, 8)
        # 取最后一个时间步的隐藏状态 h_n: (B, 8)
        _, (h_n, _) = self.qlstm(qlstm_in)
        
        # 4. 最终分类
        logits = self.classifier(h_n)
        
        return logits

# ==========================================
# 示例用法 (Sanity Check)
# ==========================================
if __name__ == "__main__":
    # 模拟 Batch=2 的数据
    B = 2
    dummy_fmri = torch.randn(B, 140, 116)
    dummy_smri = torch.randn(B, 116) # 模拟归一化后的 GMV
    dummy_adj = torch.randn(116, 116)
    dummy_adj = (dummy_adj + dummy_adj.T) / 2 # 对称化
    
    # 实例化模型
    model = Fused_Q_MGRN()
    
    # 前向传播
    output = model(dummy_fmri, dummy_smri, dummy_adj)
    
    print(f"模型输出形状: {output.shape}") # 应为 (2, 2)
    print("Fused Q-MGRN 构建成功。")