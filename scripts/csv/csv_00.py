import pandas as pd
import os

# 文件路径配置
input_csv = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/internvid.csv"
output_csv = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/internvid_00.csv"
video_dir = "/mnt/spaceai-internal/panda-intervid/internvid/nvme/tmp/heyinan/panda/00/"

# 读取CSV并生成path列
df = pd.read_csv(input_csv)

# 优化方案: 先获取目录下所有文件名集合
existing_files = set(os.listdir(video_dir))

# 筛选存在对应视频文件的行
filtered_df = df[df['path'].isin(existing_files)]

# 选择并重命名最终列
final_df = filtered_df[['path', 'Caption', 'Aesthetic_Score']]

# 保存结果
final_df.to_csv(output_csv, index=False)
