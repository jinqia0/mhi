import re
import pandas as pd
from tqdm import tqdm

def update_path(original_path):
    # 正则匹配出数字部分（如 079）
    match = re.search(r'/panda/(\d+)', original_path)
    if not match:
        return original_path  # 如果没有匹配到，返回原路径
    
    number = int(match.group(1))  # 转换为整数

    # 根据规则确定 new_range
    if 0 <= number <= 137:
        new_range = "panda-000-137"
    elif 138 <= number <= 273:
        new_range = "panda-138-273"
    elif 274 <= number <= 408:
        new_range = "panda-274-408"
    elif 409 <= number <= 440:
        new_range = "panda-409-440"
    else:
        return original_path  # 如果超出范围，返回原路径

    # 替换路径中的部分
    updated_path = re.sub(
        r'/panda\d+-\d+/',  # 匹配类似 panda076-125 的部分
        f'/{new_range}/',    # 替换为新的 range
        original_path
    )
    
    # 进一步调整路径结构（添加 /mnt/spaceai-internal/panda-intervid/untar_data/disk2/）
    final_path = f"/mnt/spaceai-internal/panda-intervid/untar_data/disk2{updated_path.split('panda-30m')[1]}"
    return final_path


if __name__ == "__main__":
    # 测试代码
    # original_path = "/workspace/public/datasets/panda-30m/panda076-125/nvme/tmp/heyinan/panda/079/JUWYUZbdRXk-0:04:19.158-0:04:30.570.mp4"
    # updated_path = update_path(original_path)
    # print(updated_path)

    csv_path = "Datasets/panda_all_text_aes.csv"
    df = pd.read_csv(csv_path)
    
    tqdm.pandas()
    df['path'] = df['path'].progress_apply(update_path)
        
    df.to_csv(csv_path, index=False)
    print(f"CSV 文件处理完成，已保存为 {csv_path}.")
