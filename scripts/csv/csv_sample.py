import pandas as pd
import shutil
import os
import random

# === 配置 ===
csv_path = "data/internvid/internvid_00_1k.csv"  # CSV 文件路径
destination_folder = "./samples"  # 目标文件夹
sample_size = 100  # 需要抽样的视频数量

# 读取 CSV
df = pd.read_csv(csv_path)

# 筛选符合条件的视频
filtered_videos = df[
    (df["caption_interaction"] == "Yes")
]

# 如果筛选结果少于 sample_size，则使用全部数据
sample_size = min(len(filtered_videos), sample_size)
sampled_videos = filtered_videos.sample(n=sample_size, random_state=42)

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 复制视频到目标文件夹
for _, row in sampled_videos.iterrows():
    video_path = os.path.join('Datasets/videos', row["path"])  # 获取视频路径
    if os.path.exists(video_path):  # 确保文件存在
        shutil.copy(video_path, destination_folder)
    else:
        print(f"⚠️ 文件未找到: {video_path}")

print(f"✅ 已成功复制 {sample_size} 个视频到 {destination_folder}")
