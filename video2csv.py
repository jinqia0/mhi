import os
import csv

# 定义根目录
root_dir = '/home/jinqiao/Projects/mhmg/Datasets/Panda/nvme/tmp/heyinan/panda/'

# 定义CSV文件的路径
csv_file_path = 'video_info_100k.csv'

# 打开CSV文件准备写入
with open(csv_file_path, mode='w+', newline='') as csv_file:
    # 定义CSV文件的列名
    fieldnames = ['video_id', 'path', 'timestamp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # 写入列名
    writer.writeheader()
    
    # 遍历根目录下的所有子目录
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        # 确保是目录
        if os.path.isdir(subdir_path):
            # 遍历子目录下的所有文件
            for file_name in os.listdir(subdir_path):
                # 检查文件是否是MP4文件
                if file_name.endswith('.mp4'):
                    # 提取video_id（固定长度为11个字符）
                    video_id = file_name[:11]  # 取前11个字符
                    
                    # 提取timestamp（从第12个字符开始到倒数第4个字符）
                    timestamp = file_name[12:-4]  # 去掉前11个字符和后4个字符（.mp4）
                    
                    # 获取文件的完整路径
                    file_path = os.path.join(subdir_path, file_name)
                    
                    # 写入CSV文件
                    writer.writerow({'video_id': video_id, 'path': file_path, 'timestamp': timestamp})

print(f"CSV文件已保存到: {csv_file_path}")