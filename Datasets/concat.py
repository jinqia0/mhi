import pandas as pd
import glob
import os

# ✅ 设置目标文件夹路径
data_dir = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/splits"

# 获取所有以 part1~part4.csv 结尾的文件
pattern = os.path.join(data_dir, "*flow.csv")
target_files = glob.glob(pattern)
print(f"找到 {len(target_files)} 个文件：{target_files}")

# 创建一个空的 DataFrame 用于存放所有拼接的数据
all_data = []

# 遍历文件路径，读取 CSV 文件并添加到 all_data 列表中
for file in target_files:
    df = pd.read_csv(file)
    all_data.append(df)

# 将所有 DataFrame 拼接成一个大的 DataFrame
final_df = pd.concat(all_data, ignore_index=True)

# 将拼接后的结果保存到一个新的 CSV 文件
output_file = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/mhi_multi_aes45_coarse_ocr5_flow.csv'
final_df.to_csv(output_file, index=False)
print(f"拼接完成，结果保存到 {output_file}")