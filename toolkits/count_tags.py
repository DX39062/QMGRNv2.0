import glob
import os

def count_research_groups():
    """
    扫描当前路径下的 .xml 文件，统计特定 <researchGroup> 标签的个数。
    """
    
    # 定义要查找的标签
    # 注意：我假设您第二个标签 <C>EMCI</researchGroup> 是一个拼写错误，
    # 意在匹配 <researchGroup>CN</researchGroup> 的格式，
    # 因此我将其更正为 <researchGroup>EMCI</researchGroup>。
    tag_cn = "<researchGroup>CN</researchGroup>"    # 标签 0
    tag_emci = "<researchGroup>EMCI</researchGroup>" # 标签 1

    # 初始化计数器
    count_cn = 0
    count_emci = 0
    
    other_files = 0
    total_xml_files = 0

    # 使用 glob 获取当前路径下所有的 .xml 文件
    xml_files = glob.glob('*.xml')
    total_xml_files = len(xml_files)

    if total_xml_files == 0:
        print("在当前路径下没有找到 .xml 文件。")
        return

    print(f"开始扫描 {total_xml_files} 个 .xml 文件...")

    for filename in xml_files:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 使用 elif 来确保一个文件只被计算一次
                # （假设一个文件只包含一种标签）
                if tag_cn in content:
                    count_cn += 1
                elif tag_emci in content:
                    count_emci += 1
                else:
                    other_files += 1
        except UnicodeDecodeError:
            print(f"警告：文件 '{filename}' 无法使用 UTF-8 编码读取，已跳过。")
        except Exception as e:
            print(f"警告：读取文件 '{filename}' 时发生错误: {e}，已跳过。")

    # 打印最终统计结果
    print("\n--- 统计结果 ---")
    print(f"标签 'CN' (0) 的文件个数: {count_cn}")
    print(f"标签 'EMCI' (1) 的文件个数: {count_emci}")
    print(f"不包含任一标签的 .xml 文件个数: {other_files}")
    print(f"总计扫描的 .xml 文件: {total_xml_files}")

if __name__ == "__main__":
    count_research_groups()