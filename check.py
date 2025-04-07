import os
import csv
from tqdm import tqdm

def validate_videos_in_csv(csv_file, video_column="path"):
    """
    验证 CSV 文件中记录的 video 文件是否可读
    
    Args:
        csv_file (str): CSV 文件路径
        video_column (str): 包含视频路径的列名（默认为 "video"）
    """
    if not os.path.exists(csv_file):
        print(f"❌ 错误：CSV 文件 '{csv_file}' 不存在！")
        return

    unreadable_videos = []
    
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        
        if video_column not in reader.fieldnames:
            print(f"❌ 错误：CSV 文件没有 '{video_column}' 列！")
            return

        # 获取总行数用于进度条
        total_rows = sum(1 for _ in reader)
        file.seek(0)  # 重置文件指针
        next(reader)  # 跳过标题行

        # 使用tqdm创建进度条
        for row in tqdm(reader, total=total_rows, desc="验证视频文件"):
            video_path = row[video_column].strip()  # 去除首尾空格
            
            if not video_path:  # 跳过空路径
                continue
            
            if not os.path.exists(video_path):
                unreadable_videos.append((video_path, "文件不存在"))
            elif not os.access(video_path, os.R_OK):
                unreadable_videos.append((video_path, "文件不可读（权限不足）"))

    # 输出结果
    if not unreadable_videos:
        print("✅ 所有 video 文件均可读！")
    else:
        print("❌ 以下 video 文件存在问题：")
        for path, error in unreadable_videos:
            print(f"- {path}: {error}")

# 示例调用
if __name__ == "__main__":
    csv_file = "Datasets/internvid_rename_abspath.csv"  # 替换为你的 CSV 文件路径
    validate_videos_in_csv(csv_file)
