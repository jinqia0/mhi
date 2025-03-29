import pandas as pd
import re
from tqdm import tqdm

def classify_panda_directory(path):
    """根据路径中的编号重新分类panda目录范围"""
    # 使用正则表达式提取编号
    match = re.search(r'/panda/(\d{3})/', path)
    if not match:
        return path  # 如果没有匹配到编号，返回原路径
    
    number = int(match.group(1))
    # print(number)
    
    # 确定新的范围目录
    if 0 <= number <= 137:
        new_range = "panda-000-137"
    elif 138 <= number <= 273:
        new_range = "panda-138-273"
    elif 274 <= number <= 408:
        new_range = "panda-274-408"
    elif 409 <= number <= 440:
        new_range = "panda-409-440"
    else:
        return path  # 如果编号不在任何范围内，返回原路径
    
    # 替换路径中的范围部分
    part = path.split('/')[6]
    
    return path.replace(part, new_range)

# 示例使用
if __name__ == "__main__":
    
    # # 示例路径
    # example_path = "/mnt/spaceai-internal/panda-intervid/untar_data/disk2/panda076-125/nvme/tmp/heyinan/panda/079/JUWYUZbdRXk-0:04:19.158-0:04:30.570.mp4"
    # new_path = classify_panda_directory(example_path)
    # print(f"原路径: {example_path}")
    # print(f"新路径: {new_path}")
    
    # 实际应用中，你可以这样处理整个DataFrame
    # 假设df是你的DataFrame，包含'path'列
    df = pd.read_csv('/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/mhi.csv')
    tqdm.pandas(desc="Processing")
    df['path'] = df['path'].apply(classify_panda_directory)
    
    df.to_csv('/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/mhi.csv', index=False)
    print("处理完成，结果已保存到 /mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/mhi.csv")
    