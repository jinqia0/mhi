import pandas as pd

# 读取 CSV 文件
csv_path = "./panda_10k_interaction_score.csv"
df = pd.read_csv(csv_path)

# 1. 筛选符合条件的视频：num_persons >= 2 且 interaction_score == 5
filtered_videos = df[(df["num_persons"] >= 2) & (df["interaction_score"] == 5)]

# 2. 统计 has_interaction 为 1 的比例
total_videos = len(filtered_videos)
if total_videos > 0:
    interaction_videos = filtered_videos[filtered_videos["has_interaction"] == 1]
    interaction_ratio = len(interaction_videos) / total_videos
else:
    interaction_ratio = 0

# 打印比例
print(f"在 `num_persons >= 2 且 interaction_score == 5` 的视频中，has_interaction 为 1 的比例: {interaction_ratio:.4f}")
