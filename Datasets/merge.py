import pandas as pd

csv_path = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/panda/panda_multi_aes45_part2.csv'
coarse_path = csv_path.replace('.csv', '_coarse.csv')

df1 = pd.read_csv(csv_path)
df2 = pd.read_csv(coarse_path)

# 清除列名空格
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# 合并，确保 path 保留
df2_other_cols = [col for col in df2.columns if col != 'path']
merged_df = df1.merge(df2[['path'] + df2_other_cols], on='path', how='left')

# 保存结果
merged_df.to_csv(csv_path, index=False)
