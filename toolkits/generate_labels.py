import glob
import xml.etree.ElementTree as ET

# 1. 定义从 XML 标签到数字标签的映射
label_map = {
    'EMCI': 1,
    'CN': 0
}

# 字典，用于存储最终结果
patient_labels = {}

# 2. 查找当前目录下的所有 .xml 文件
print("正在扫描当前目录中的 .xml 文件...")
xml_files = glob.glob('*.xml')

if not xml_files:
    print("错误：在当前目录中未找到 .xml 文件。")
else:
    print(f"找到了 {len(xml_files)} 个 .xml 文件。正在处理...")

# 3. 循环遍历每个文件并解析
for xml_file in xml_files:
    try:
        # 解析 XML 文件
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 4. 查找所需的标签 (使用 .// 可以在 XML 树的任何深度搜索)
        subject_element = root.find('.//subjectIdentifier')
        group_element = root.find('.//researchGroup')
        
        # 5. 检查是否找到了两个标签
        if subject_element is not None and group_element is not None:
            subject_id = subject_element.text
            group = group_element.text
            
            # 6. 检查 researchGroup 是否是我们关心的 'EMCI' 或 'CN'
            if group in label_map:
                label = label_map[group]
                # 将结果存入字典
                patient_labels[subject_id] = label
            else:
                # 可选：取消注释以查看被跳过的文件
                # print(f"跳过 {xml_file}: 组为 '{group}' (不是 EMCI 或 CN)。")
                pass
        else:
            print(f"跳过 {xml_file}: 未能找到 <subjectIdentifier> 或 <researchGroup> 标签。")
            
    except ET.ParseError:
        print(f"跳过 {xml_file}: 不是一个有效的 XML 文件。")
    except Exception as e:
        print(f"处理 {xml_file} 时发生错误: {e}")

# 7. 按照您期望的格式打印最终输出
print("\n--- 标签输出 ---")
if not patient_labels:
    print("未找到 'EMCI' 或 'CN' 组的参与者。")
else:
    for subject, label in patient_labels.items():
        print(f"'{subject}': {label},")