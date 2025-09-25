import dask.dataframe as dd
import os
from dask.diagnostics import ProgressBar

# 读取文件 B，并提取文件名
file_B = dd.read_csv('Datasets/mhi/mhi_multi_aes48_coarse_ocr5_flow.csv', usecols=['path'])
file_B['filename'] = file_B['path'].apply(lambda x: os.path.basename(x), meta=('x', 'str'))
file_B['filename'] = file_B['filename'].astype(str)

# 读取文件 A，Dask 会自动分块处理
file_A = dd.read_csv('file_scan_output/merged_file.csv', usecols=['filename', 'filepath'])
file_A['filename'] = file_A['filename'].astype(str)
file_A['filepath'] = file_A['filepath'].astype(str)

# 合并 file_B 和 file_A
merged_df = dd.merge(file_B, file_A, on='filename', how='inner')

# 使用 Dask 的内置进度条来监视计算过程
with ProgressBar():
    merged_df = merged_df.compute()  # 触发实际计算，Dask 通过惰性计算执行操作
    
# 保存最终结果到 CSV 文件
output_path = 'file_scan_output/paths_2.8M_dask.csv'
merged_df.to_csv(output_path, index=False, single_file=True)

# 输出结果
print(f'Saved merged paths to {output_path}, total {len(merged_df)} rows.')
