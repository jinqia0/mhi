import pandas as pd

# 输入 CSV 文件路径
csv_path = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/internvid/internvid_00_1k.csv"

# 读取 CSV
df = pd.read_csv(csv_path)

# 确保 'path' 列存在
if 'path' not in df.columns:
    raise ValueError("CSV 文件缺少 'path' 列，请检查文件内容！")

# 替换路径前缀
old_prefix = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/videos"
new_prefix = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/videos/"

df['path'] = df['path'].apply(lambda x: x.replace(old_prefix, new_prefix) if isinstance(x, str) else x)

# 保存回 CSV 文件
df.to_csv(csv_path, index=False)

print(f"已更新 'path' 列的路径前缀，保存到 {csv_path}")
