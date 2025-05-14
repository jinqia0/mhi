import pandas as pd

csv_path = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/mhi_multi_aes48_coarse_ocr5_flow.csv'

# 读取 CSV 文件（Dask 会处理大的文件并进行并行计算）
df = pd.read_csv(csv_path)

# 筛选出 num_humans >= 2 的行
filtered_df = df[df['max_iou'] > 0]

# 将结果保存为新的 CSV 文件
output_path = csv_path.replace('.csv', '_iou.csv')
filtered_df.to_csv(output_path, index=False)

print(f"提取完成({len(filtered_df)}/{len(df.shape)})，结果保存在 {output_path}")
