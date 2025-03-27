import pandas as pd
import os
import re

def get_disk_group(panda_range):
    """根据panda范围确定磁盘分组"""
    # 提取起始编号
    start_num = int(re.search(r'panda(\d+)-', panda_range).group(1))
    
    if 0 <= start_num <= 137:
        return "panda-000-137"
    elif 138 <= start_num <= 273:
        return "panda-138-273"
    elif 274 <= start_num <= 408:
        return "panda-274-408"
    elif 409 <= start_num <= 440:
        return "panda-409-440"
    else:
        raise ValueError(f"无法根据起始编号{start_num}确定panda范围")

def convert_path(old_path):
    """将旧路径转换为新路径格式"""
    # 分解旧路径
    parts = old_path.split('/')
    
    # 找到pandaXXX-XXX部分
    panda_range = next((p for p in parts if p.startswith('panda') and re.match(r'panda\d+-\d+', p)), None)
    
    if not panda_range:
        raise ValueError(f"无法从路径中提取panda范围: {old_path}")
    
    # 确定磁盘分组
    disk_group = get_disk_group(panda_range)
    
    # 提取关键部分（从pandaXXX-XXX开始的部分）
    key_parts = []
    found = False
    for part in parts:
        if part == panda_range:
            found = True
        if found:
            key_parts.append(part)
    
    # 构建新路径
    new_path = os.path.join('/mnt/spaceai-internal/panda-intervid/untar_data/disk2', disk_group, *key_parts)
    return new_path

def process_csv(input_csv, output_csv):
    """处理CSV文件并保存结果"""
    df = pd.read_csv(input_csv)
    
    # 检查是否存在path列
    if 'path' not in df.columns:
        raise ValueError("CSV文件中没有找到'path'列")
    
    # 应用路径转换
    df['path'] = df['path'].apply(convert_path)
    
    # 保存结果
    df.to_csv(output_csv, index=False)
    print(f"处理完成，结果已保存到: {output_csv}")

# 使用示例
input_csv = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/panda_all_text_aes.csv"  # 原始CSV文件路径
output_csv = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/mhi.csv"  # 处理后的CSV文件保存路径

process_csv(input_csv, output_csv)
