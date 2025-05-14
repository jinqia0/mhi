import pandas as pd

# 读取原始文件
csv_path = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/panda/panda_multi_aes45_part2.csv'
df = pd.read_csv(csv_path)

# 去除列名空格（可选但推荐）
df.columns = df.columns.str.strip()

# 仅保留指定列
keep_cols = ['path', 'text', 'aes', 'has_person', 'num_persons', 'bbox_overlap', 'new_aes']
df_filtered = df[keep_cols]

# 保存结果到新文件（避免覆盖原始文件）
df_filtered.to_csv(csv_path, index=False)
