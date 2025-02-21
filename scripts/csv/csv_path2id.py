import pandas as pd
import os

csv_file = "./data/panda_10k_interaction_score.csv"

# 读取原始 CSV 文件
df = pd.read_csv(csv_file)

# 修改 'path' 列，保留倒数第二级和最后一级目录
def modify_path(path):
    # 获取倒数第二级和最后一级目录
    parts = path.split(os.sep)
    return os.path.join(parts[-2], parts[-1])

df['path'] = df['path'].apply(modify_path)

# 将结果保存为新的 CSV 文件
df.to_csv(csv_file, index=False)

print(f"CSV 文件处理完成，已保存为 {csv_file}.")
