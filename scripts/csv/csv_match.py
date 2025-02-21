import pandas as pd
import os

def read_filenames(filepath):
    with open(filepath, 'r') as file:
        return [line.strip() for line in file if line.strip()]

panda1_files = read_filenames('HOI-multiperson/panda1.txt')  
panda2_files = read_filenames('HOI-multiperson/panda2.txt')

# 合并 panda1 和 panda2 的文件名列表
all_files = set(panda1_files + panda2_files)
print(len(all_files))

# # 读取 panda_10k_interaction_score.csv 文件
# df = pd.read_csv('data/panda_10k_interaction_score.csv')

# # 提取路径中的文件名
# df['filename'] = df['path'].apply(lambda x: x.split(os.sep)[-1])

# # 新增 inHOI 列并标记
# df['inHOI'] = df['filename'].apply(lambda x: 1 if x[:-4] in all_files else 0)

# # 保存结果到新的 CSV 文件
# df.to_csv('data/panda_10k_interaction_score.csv', index=False)