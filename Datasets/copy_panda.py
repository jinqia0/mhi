import pandas as pd
import shutil
import os
from multiprocessing import Pool
from tqdm import tqdm

# 定义复制文件的函数
def copy_video_file(source_path, destination_folder):
    try:
        if os.path.exists(source_path):  # 检查源文件是否存在
            # 获取视频文件的文件名
            file_name = os.path.basename(source_path)
            # 构建目标文件的路径
            destination_path = os.path.join(destination_folder, file_name)

            # 复制文件
            shutil.copy(source_path, destination_path)
            return file_name  # 返回已复制的文件名
        else:
            return None  # 如果文件不存在
    except Exception as e:
        print(f"处理文件 {source_path} 时出错: {e}")
        return None

# 将复制文件的函数包装为全局可 picklable
def copy_video_file_wrapper(args):
    source_path, destination_folder = args
    return copy_video_file(source_path, destination_folder)

# 定义处理 CSV 文件的函数
def process_csv(csv_file, destination_folder):
    try:
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)

        # 提取所有源文件路径
        source_paths = df['path'].tolist()  # 假设路径列为 'path'

        # 使用多进程复制文件并显示进度条
        with Pool(processes=64) as pool:
            # 使用 tqdm 包装进度条
            results = list(tqdm(pool.imap(copy_video_file_wrapper, [(source_path, destination_folder) for source_path in source_paths]), 
                                total=len(source_paths),  # 进度条的总数
                                desc="复制文件",  # 进度条描述
                                unit="文件"))  # 每个进度条单元

            # 过滤掉 None（表示文件不存在的情况）
            successful_copies = [result for result in results if result is not None]
            print(f"成功复制 {len(successful_copies)} 个文件")
    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {e}")


# 指定目标文件夹
destination_folder = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Data/panda/19-23'

# 创建目标文件夹，如果不存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 处理第一部分的 CSV 文件
csv_file = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/panda/panda_19-23_multi.csv'

# 执行处理任务
process_csv(csv_file, destination_folder)
