import os
import pandas as pd
from tqdm import tqdm

# 读取 CSV 文件
csv_file = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/panda/panda_19-23_multi_aes.csv'  # 替换为你的 CSV 文件路径
df = pd.read_csv(csv_file)

# 提取 new_aes 大于 4.5 的行
df_filtered = df[df['new_aes'] > 4.5]

# 遍历 new_aes 小于等于 4.5 的行，删除对应的文件
df_to_delete = df[df['new_aes'] <= 4.5]

pbar = tqdm(total=len(df_to_delete), desc="删除文件进度", unit="个文件")
for index, row in df_to_delete.iterrows():
    video_path = row['path']
    try:
        # 删除文件
        os.remove(video_path)
    except Exception as e:
        print(f"删除文件 {video_path} 时出错: {e}")
    pbar.update(1)
pbar.close()

# 如果需要，将筛选后的结果保存为新的 CSV 文件
output_file = csv_file.replace('aes', 'aes45') # 替换为你的输出文件路径
df_filtered.to_csv(output_file, index=False)
print(f"筛选后的数据已保存为 {output_file}")