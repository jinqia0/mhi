import pandas as pd
import os

# 原始文件路径
input_csv = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/mhi_multi_aes48_coarse_ocr5_flow_iou.csv"
# 指定输出文件夹
output_dir = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/new_splits"
# 每个子文件的最大行数（不含列名）
chunk_size = 200_000

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 使用分块读取
reader = pd.read_csv(input_csv, chunksize=chunk_size)

# 保存分块
for i, chunk in enumerate(reader):
    base_name = os.path.basename(input_csv).rsplit('.csv', 1)[0]
    output_csv = os.path.join(output_dir, f"{base_name}_part_{str(i+1).zfill(2)}.csv")
    chunk.to_csv(output_csv, index=False)
    print(f"已保存: {output_csv}")
