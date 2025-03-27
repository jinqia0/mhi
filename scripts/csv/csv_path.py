import pandas as pd

# 读取CSV文件
df = pd.read_csv("/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/internvid.csv")

# 生成path列
df['path'] = df['YoutubeID'] + '-' + df['Start_timestamp'] + '-' + df['End_timestamp'] + '.mp4'

# 选择需要的列并重命名
new_df = df[['path', 'Caption', 'Aesthetic_Score']]

# 保存新的CSV文件
new_df.to_csv("/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/internvid.csv", index=False)
