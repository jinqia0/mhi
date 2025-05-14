import os
import csv
from tqdm import tqdm

def validate_videos_in_csv(csv_file, video_column="path", output_csv="unreadable_videos.csv"):
    """
    验证 CSV 文件中记录的 video 文件是否可读，并将有问题的文件保存到新的 CSV 中
    
    Args:
        csv_file (str): CSV 文件路径
        video_column (str): 包含视频路径的列名（默认为 "path"）
        output_csv (str): 输出有问题视频的 CSV 文件路径（默认为 "unreadable_videos.csv"）
    """
    if not os.path.exists(csv_file):
        print(f"❌ 错误：CSV 文件 '{csv_file}' 不存在！")
        return

    unreadable_videos = []
    original_fieldnames = None
    
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        original_fieldnames = reader.fieldnames
        
        if video_column not in original_fieldnames:
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
                row['error'] = "文件不存在"
                unreadable_videos.append(row)
            elif not os.access(video_path, os.R_OK):
                row['error'] = "文件不可读（权限不足）"
                unreadable_videos.append(row)

    # 输出结果
    if not unreadable_videos:
        print("✅ 所有 video 文件均可读！")
        return
    
    print(f"❌ 发现 {len(unreadable_videos)} 个有问题的视频文件，已保存到 {output_csv}")
    
    # 将有问题视频写入新的CSV文件
    with open(output_csv, mode="w", encoding="utf-8", newline='') as out_file:
        # 添加error列到原始字段名
        fieldnames = original_fieldnames + ['error']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(unreadable_videos)

# 示例调用
if __name__ == "__main__":
    csv_file = "Datasets/internvid_rename_abspath.csv"  # 替换为你的 CSV 文件路径
    validate_videos_in_csv(csv_file)
