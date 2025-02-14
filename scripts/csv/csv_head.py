import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('./panda_10k_interaction_score.csv')

# 提取前100行
df_top_100 = df.head(1000)

# 保存为新的CSV文件
df_top_100.to_csv('./panda_1k.csv', index=False)