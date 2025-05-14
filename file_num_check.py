import os

def count_files_in_subfolders(directory):
    file_count = {}
    
    # 遍历给定目录中的所有子文件夹
    for subdir, dirs, files in os.walk(directory):
        # 只统计子文件夹中的文件，不统计根目录
        if subdir != directory:
            file_count[subdir] = len(files)
    
    return file_count

# 示例使用
directory = '/mnt/spaceai-internal/panda-intervid/untar_data/disk2/panda-000-137/nvme/tmp/heyinan/panda'  # 替换为你想要检查的目录路径
file_count = count_files_in_subfolders(directory)

# 输出每个子文件夹的文件数
for subfolder, count in file_count.items():
    print(f"子文件夹: {subfolder} 中有 {count} 个文件")
