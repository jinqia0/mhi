import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/panda/panda_10k_interaction_score.csv')  

count_y = df['caption_interaction'] == 'Yes'
count_n = df['caption_interaction'] == 'No'

# 打印不同条件的行数
print(f"yes: {count_y.sum()}")
print(f"no: {count_n.sum()}")
print(f"既非yes也非no: {(~count_y & ~count_n).sum()}")
print(f"总行数: {len(df)}")

# 打印既非yes也非no的行下标
print(f"既非yes也非no的行下标: {df[~count_y & ~count_n].index.to_list()}")

