import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('data/internvid/internvid_00_1k.csv')

# 提取前100行
df_top_100 = df.head(100)

# 保存为新的CSV文件
df_top_100.to_csv('data/internvid/internvid_00_100.csv', index=False)