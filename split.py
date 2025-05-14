import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('input_file.csv')

# 计算每个拆分文件的行数
split_size = len(df) // 3

# 拆分文件并保存
df.iloc[:split_size].to_csv('split_1.csv', index=False)
df.iloc[split_size:2*split_size].to_csv('split_2.csv', index=False)
df.iloc[2*split_size:].to_csv('split_3.csv', index=False)
