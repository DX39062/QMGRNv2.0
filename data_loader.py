import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ==========================================
# 第一部分: 静态图结构加载器
# ==========================================

def load_adjacency_matrix(adj_file, n_nodes=116):
    """
    加载静态的功能连接矩阵 (FC.csv) 并转换为 PyTorch 张量
    """
    print(f"--- 正在加载邻接矩阵: {adj_file} ---")
    try:
        # 读取 CSV (假设无表头，纯数字矩阵)
        adj_np = pd.read_csv(adj_file, header=None).values.astype(np.float32)
        
        # 检查维度
        if adj_np.shape != (n_nodes, n_nodes):
            raise ValueError(f"邻接矩阵形状错误: 期望 ({n_nodes}, {n_nodes}), 实际 {adj_np.shape}")
            
        # 预处理: 将对角线设为 0 (移除自环，避免 inf 问题)
        np.fill_diagonal(adj_np, 0)
        
        # 转换为 Tensor
        adj_torch = torch.tensor(adj_np)
        
        # 归一化 (Symmetric Normalization) - GCN 的标准步骤
        # D^-0.5 * A * D^-0.5
        # 1. 添加自环 (A + I)
        A_tilde = adj_torch + torch.eye(n_nodes)
        # 2. 计算度矩阵
        degrees = torch.sum(A_tilde, dim=1)
        D_inv_sqrt = torch.pow(degrees, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0 # 处理除零
        D_mat = torch.diag(D_inv_sqrt)
        # 3. 归一化
        adj_normalized = D_mat @ A_tilde @ D_mat
        
        print("邻接矩阵加载并归一化完成。")
        return adj_normalized

    except Exception as e:
        print(f"加载邻接矩阵时出错: {e}")
        raise e

# ==========================================
# 第二部分: 多模态数据集类 (PyTorch Dataset)
# ==========================================

class MultimodalDataset(Dataset):
    def __init__(self, fmri_dir, smri_dir, label_file, n_time_steps=140, n_nodes=116):
        """
        参数:
        fmri_dir: fMRI 时序数据文件夹路径 (tsdatasets/)
        smri_dir: sMRI 特征数据文件夹路径 (sMRI_features/)
        label_file: 标签文件路径 (labels.csv)
        """
        self.fmri_dir = fmri_dir
        self.smri_dir = smri_dir
        self.n_time_steps = n_time_steps
        self.n_nodes = n_nodes
        
        # 1. 解析 labels.csv
        # 格式期望: 'id': label (例如 '129_S_6228': 0)
        print(f"--- 正在解析标签文件: {label_file} ---")
        self.label_map = {}
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if ':' in line:
                        parts = line.strip().split(':')
                        # 清理 ID: 去除单引号和空格 -> 129_S_6228
                        clean_id = parts[0].strip().replace("'", "")
                        # 清理 Label: 去除逗号 -> 0
                        clean_label = int(parts[1].strip().replace(",", ""))
                        self.label_map[clean_id] = clean_label
        except Exception as e:
            print(f"解析标签文件失败: {e}")
            raise e
            
        # 2. 扫描文件夹并匹配数据
        self.data_list = [] # 存储 (fmri_path, smri_path, label)
        
        # 获取所有 fMRI 文件
        if not os.path.exists(fmri_dir):
            raise FileNotFoundError(f"找不到 fMRI 文件夹: {fmri_dir}")
            
        fmri_files = sorted([f for f in os.listdir(fmri_dir) if f.endswith('.csv')])
        
        print(f"正在匹配多模态数据 (fMRI + sMRI)...")
        match_count = 0
        
        for f_file in fmri_files:
            # fMRI 文件名格式: 129_S_6228_20180220.csv
            # 提取 ID (前三部分): 129_S_6228
            file_id = '_'.join(f_file.split('_')[:3])
            
            # A. 检查是否有标签
            if file_id in self.label_map:
                label = self.label_map[file_id]
                
                # B. 检查是否有对应的 sMRI 文件
                # sMRI 文件名格式: ID.csv (例如 129_S_6228.csv)
                s_file = f"{file_id}.csv"
                s_path = os.path.join(smri_dir, s_file)
                
                if os.path.exists(s_path):
                    f_path = os.path.join(fmri_dir, f_file)
                    self.data_list.append((f_path, s_path, label))
                    match_count += 1
                else:
                    # 可选: 打印缺失 sMRI 的警告
                    # print(f"警告: ID {file_id} 缺少 sMRI 文件，已跳过。")
                    pass
            else:
                # 可选: 打印缺失标签的警告
                pass
                
        print(f"匹配完成: 找到 {match_count} 个完整的多模态样本。")
        if match_count == 0:
            raise RuntimeError("未找到任何匹配的数据样本！请检查文件名和 IDs 是否一致。")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        加载并返回一个样本的数据
        """
        f_path, s_path, label = self.data_list[idx]
        
        try:
            # 1. 加载 fMRI 数据 (140, 116)
            fmri_data = pd.read_csv(f_path, header=None).values.astype(np.float32)
            
            # 形状检查
            if fmri_data.shape != (self.n_time_steps, self.n_nodes):
                # 如果形状不对，尝试转置 (有些工具可能会保存为 116x140)
                if fmri_data.shape == (self.n_nodes, self.n_time_steps):
                    fmri_data = fmri_data.T
                else:
                    raise ValueError(f"fMRI 形状错误: {f_path}, {fmri_data.shape}")

            # fMRI 标准化 (Z-score normalization per subject)
            # 这对神经网络和量子电路都至关重要
            scaler = StandardScaler()
            fmri_data = scaler.fit_transform(fmri_data) # 对每个 ROI (列) 进行标准化

            # 2. 加载 sMRI 数据 (116,)
            smri_data = pd.read_csv(s_path, header=None).values.astype(np.float32)
            smri_data = smri_data.flatten() # 确保是 1D
            
            # sMRI 形状检查
            if smri_data.shape[0] != self.n_nodes:
                raise ValueError(f"sMRI 特征维度错误: {s_path}, {smri_data.shape}")
            
            # sMRI 标准化 (简单的归一化，因为它是 GMV)
            # 避免数值过大导致神经网络饱和
            # 这里使用简单的除以最大值或 Z-score
            if np.std(smri_data) > 0:
                smri_data = (smri_data - np.mean(smri_data)) / np.std(smri_data)
            else:
                smri_data = smri_data - np.mean(smri_data)

            # 3. 返回 Tensor
            return (
                torch.tensor(fmri_data),  # (140, 116)
                torch.tensor(smri_data),  # (116,)
                torch.tensor(label, dtype=torch.long)
            )
            
        except Exception as e:
            print(f"读取数据出错 (索引 {idx}): {e}")
            # 返回全零数据防止 Crash (或者直接抛出异常)
            return (torch.zeros(self.n_time_steps, self.n_nodes), 
                    torch.zeros(self.n_nodes), 
                    torch.tensor(0, dtype=torch.long))

# ==========================================
# 第三部分: 使用示例 (集成到训练脚本)
# ==========================================
if __name__ == "__main__":
    # 配置路径
    base_path = "./"  # 您的项目根目录
    fmri_folder = os.path.join(base_path, "tsdatasets")
    smri_folder = os.path.join(base_path, "sMRI_features")
    labels_file = os.path.join(base_path, "labels.csv")
    adj_file = os.path.join(base_path, "FC.csv")
    
    # 1. 加载邻接矩阵 (只需一次)
    if os.path.exists(adj_file):
        adj_matrix = load_adjacency_matrix(adj_file)
        print(f"Adj shape: {adj_matrix.shape}")
    else:
        print("警告: FC.csv 未找到，请确保文件存在。")
    
    # 2. 创建数据集
    if os.path.exists(fmri_folder) and os.path.exists(smri_folder) and os.path.exists(labels_file):
        dataset = MultimodalDataset(fmri_folder, smri_folder, labels_file)
        
        # 测试 DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 获取一个 Batch 看看
        for fmri_batch, smri_batch, label_batch in loader:
            print("\n--- 数据加载成功 ---")
            print(f"fMRI Batch: {fmri_batch.shape} (Batch, Time, Nodes)")
            print(f"sMRI Batch: {smri_batch.shape} (Batch, Nodes)")
            print(f"Labels: {label_batch}")
            break
    else:
        print("未找到数据文件夹，跳过数据集测试。")