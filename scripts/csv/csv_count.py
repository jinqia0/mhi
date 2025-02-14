import pandas as pd

# 读取包含判定结果的 CSV 文件
csv_file = "./panda_10k_interaction_score.csv"  # 替换为你的文件路径
df = pd.read_csv(csv_file)

# 统计每个层级的 "Yes" 和 "No"
statistics = {}

yes_count = df['caption_interaction'].value_counts().get("Yes", 0)
no_count = df['caption_interaction'].value_counts().get("No", 0)

# 输出统计结果
print(f"  Yes: {yes_count}")
print(f"  No: {no_count}")
