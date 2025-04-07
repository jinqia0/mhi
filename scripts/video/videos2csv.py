import os
import re
import pandas as pd

# 配置路径
video_dir = "Datasets/videos/"
input_csv = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/internvid/internvid.csv"
output_csv = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/internvid/internvid_00.csv"

# 1. 提取视频文件信息
video_entries = set()
pattern = r"^(.*?)-(\d{2}:\d{2}:\d{2}\.\d{3})-(\d{2}:\d{2}:\d{2}\.\d{3})\.mp4$"

for filename in os.listdir(video_dir):
    match = re.match(pattern, filename)
    if match:
        yt_id = match.group(1)
        start = match.group(2)
        end = match.group(3)
        video_entries.add((yt_id, start, end))

# 2. 读取并处理CSV
df = pd.read_csv(input_csv)
df["Key"] = list(zip(df["YoutubeID"], df["Start_timestamp"], df["End_timestamp"]))

# 筛选存在的条目
filtered_df = df[df["Key"].isin(video_entries)].copy()

# 3. 生成路径并保存
filtered_df["path"] = (
    filtered_df["YoutubeID"] + "-" +
    filtered_df["Start_timestamp"] + "-" +
    filtered_df["End_timestamp"] + ".mp4"
)
filtered_df[["path", "Caption", "Aesthetic_Score"]].to_csv(output_csv, index=False)

print(f"处理完成！生成文件：{output_csv}")
