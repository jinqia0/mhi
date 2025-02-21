import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./data/panda_10k_interaction_score.csv')  

# 统计满足某个条件的行数，例如筛选 "age" 大于 30 的行
count_both = df[(df['has_interaction'] == 1) & (df['inHOI'] == 1)].shape[0]
count_interaction = df[df['has_interaction'] == 1].shape[0]
count_HOI = df[df['inHOI'] == 1].shape[0]


print(f"count_both: {count_both}")
print(f"count_interaction: {count_interaction}")
print(f"count_HOI: {count_HOI}")
