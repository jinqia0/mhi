import pandas as pd
import os

# 读取原始 CSV 文件
df = pd.read_csv('/home/jinqiao/Projects/mhi/Datasets/Panda/panda_all_text_aes.csv')

# 筛选出包含 'nvme/tmp/heyinan/panda/000' 或 'nvme/tmp/heyinan/panda/001' 的行
filtered_df = df[df['path'].str.contains('nvme/tmp/heyinan/panda/000|nvme/tmp/heyinan/panda/001')]

# 修改 'path' 列，保留倒数第二级和最后一级目录
def modify_path(path):
    # 获取倒数第二级和最后一级目录
    parts = path.split(os.sep)
    return os.path.join(parts[-2], parts[-1])

filtered_df['path'] = filtered_df['path'].apply(modify_path)

# 将结果保存为新的 CSV 文件
filtered_df.to_csv('data/panda_100k.csv', index=False)

print("CSV 文件处理完成，已保存为 'filtered_file.csv'.")
