import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

# 配置
csv_path = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/internvid_00-15.csv'
base_dir = '/mnt/spaceai-internal/panda-intervid/internvid/nvme/tmp/heyinan/panda'
output_csv = 'Datasets/internvid_rename_abspath.csv'
num_workers = 16  # 根据CPU核心数调整

print("开始建立文件索引...")
start_time = time.time()

# 建立文件名到路径的映射字典
file_index = {}

def index_files(subdir):
    dir_path = os.path.join(base_dir, f"{subdir:02d}")
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_index[file] = os.path.join(root, file)

# 使用多线程建立索引（00-82共83个目录）
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(index_files, range(0, 83))

print(f"索引建立完成，耗时 {time.time()-start_time:.2f} 秒")
print(f"已索引 {len(file_index)} 个文件")

# 处理CSV
print("开始处理CSV文件...")
df = pd.read_csv(csv_path)

def get_absolute_path(filename):
    return file_index.get(filename, f"NOT_FOUND_{filename}")

df['path'] = df['path'].apply(get_absolute_path)

# 保存结果
df.to_csv(output_csv, index=False)
print(f"处理完成，结果已保存到 {output_csv}")
