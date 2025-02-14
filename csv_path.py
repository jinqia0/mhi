import pandas as pd

# 读取 CSV 文件
csv_file = "./panda_10k_results.csv"  # 你的 CSV 文件
df = pd.read_csv(csv_file)

# 替换路径，使其变为相对路径
df["path"] = df["path"].str.replace(r".*?/Datasets/", "Datasets/", regex=True)

# 直接覆盖原文件
df.to_csv(csv_file, index=False)

print(f"路径已更新，文件已覆盖：{csv_file}")
