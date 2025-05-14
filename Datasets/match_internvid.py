import os
import dask.dataframe as dd
from tqdm import tqdm  # 引入 tqdm

# 定义文件路径
folder_path = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/internvid/76-82/'
csv_file_path = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/internvid_rename_col.csv'
output_csv_path = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/internvid_76-82.csv'

# 读取 CSV 文件（使用 dask）
df = dd.read_csv(csv_file_path)

# 获取子文件夹中的文件名及其完整路径，创建文件名与完整路径的字典
file_paths_dict = {}

# 使用 tqdm 显示进度条
for root, dirs, files in tqdm(os.walk(folder_path), desc="扫描文件夹", unit="文件夹"):
    for file in files:
        file_paths_dict[file] = os.path.join(root, file)

# 筛选出路径列与文件名相匹配的行
matching_rows = df[df['path'].isin(file_paths_dict.keys())]

# 使用文件名从字典中获取完整路径，并更新匹配行的 'path' 列
matching_rows['path'] = matching_rows['path'].map(file_paths_dict)

# 将匹配的行保存到新的 CSV 文件（使用 dask 的 to_csv 保存大文件）
matching_rows.to_csv(output_csv_path, index=False, single_file=True)

print(f"筛选出的行已保存到 {output_csv_path}")
