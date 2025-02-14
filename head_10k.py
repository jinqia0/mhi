import pandas as pd

# 读取 CSV 文件
input_csv_path = "./panda_10k_interaction_score.csv"  # 替换为你的文件路径
df = pd.read_csv(input_csv_path)

# 提取前 10,000 行
df_first_10000 = df.head(10000)

# 保存提取的数据到新的 CSV 文件
output_csv_path = "panda_10k.csv"
df_first_10000.to_csv(output_csv_path, index=False)

print(f"前 10,000 行已保存到: {output_csv_path}")