import pandas as pd

# 输入 CSV 文件路径
csv_path = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/internvid_00.csv"

# 读取 CSV
df = pd.read_csv(csv_path)

# 确保 'path' 列存在
if 'path' not in df.columns:
    raise ValueError("CSV 文件缺少 'path' 列，请检查文件内容！")

# # 替换路径前缀
# old_prefix = "/mnt/spaceai-internal/panda-intervid/internvid/nvme/tmp/heyinan/panda/00"
# new_prefix = "/mnt/spaceai-internal/panda-intervid/internvid/nvme/tmp/heyinan/panda/00/"

# df['path'] = df['path'].apply(lambda x: x.replace(old_prefix, new_prefix) if isinstance(x, str) else x)

df['path'] = "/mnt/spaceai-internal/panda-intervid/internvid/nvme/tmp/heyinan/panda/00/" + df['path']

# 保存回 CSV 文件
df.to_csv(csv_path, index=False)

print(f"已更新 'path' 列的路径前缀，保存到 {csv_path}")
