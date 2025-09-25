#!/usr/bin/env python3
import subprocess
import sys
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 配置日志
log_file = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/scripts/upload.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def upload_file(file_path, config_dir, remote_dir, log_file):
    """使用bypy上传单个文件，并指定配置目录和远程目录"""
    if not os.path.exists(file_path):
        msg = f"文件不存在: {file_path}"
        print(msg)
        logging.warning(msg)
        return False
        
    # 构造bypy命令，指定配置目录和远程目录
    cmd = ['bypy', '--config-dir', config_dir, 'upload', file_path, remote_dir]
    
    try:
        msg = f"正在上传: {file_path} 到 {remote_dir}"
        print(msg)
        logging.info(msg)
        
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        msg = f"上传成功: {file_path} 到 {remote_dir} (耗时: {end_time - start_time:.2f}秒)"
        print(msg)
        logging.info(msg)
        logging.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        msg = f"上传失败: {file_path} 到 {remote_dir}\n错误信息: {e.stderr}"
        print(msg)
        logging.error(msg)
        return False
    except subprocess.TimeoutExpired:
        msg = f"上传超时: {file_path} 到 {remote_dir}"
        print(msg)
        logging.warning(msg)
        return False

def upload_files_in_batches(file_list_path, config_dir, remote_dir, batch_size=100000, max_workers=5):
    """分批上传文件到指定远程目录"""
    # 读取文件列表
    with open(file_list_path, 'r') as f:
        # 跳过第一行（标题行）
        file_paths = [line.strip() for line in f.readlines()[1:]]
    
    total_files = len(file_paths)
    print(f"总文件数: {total_files}")
    logging.info(f"总文件数: {total_files}")
    
    # 分批处理
    for i in range(0, total_files, batch_size):
        batch_files = file_paths[i:i+batch_size]
        batch_number = i // batch_size + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        msg = f"开始上传批次 {batch_number}/{total_batches}，包含 {len(batch_files)} 个文件到 {remote_dir}"
        print(msg)
        logging.info(msg)
        
        # 并行上传
        successful_uploads = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(upload_file, file_path, config_dir, remote_dir, log_file): file_path 
                for file_path in batch_files
            }
            
            # 处理完成的任务
            for j, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        successful_uploads += 1
                except Exception as e:
                    msg = f"上传过程中出现异常: {file_path} - {str(e)}"
                    print(msg)
                    logging.error(msg)
                
                # 显示进度
                progress = (j + 1) / len(batch_files) * 100
                msg = f"批次 {batch_number} 进度: {progress:.1f}% ({j+1}/{len(batch_files)})"
                print(msg)
                logging.info(msg)
        
        msg = f"批次 {batch_number} 完成，成功上传 {successful_uploads}/{len(batch_files)} 个文件到 {remote_dir}"
        print(msg)
        logging.info(msg)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python3 upload_enhanced.py <file_list_path> <config_dir> <remote_dir>")
        sys.exit(1)
    
    file_list_path = sys.argv[1]
    config_dir = sys.argv[2]
    remote_dir = sys.argv[3]
    
    # 在后台运行
    pid = os.fork()
    if pid > 0:
        print(f"上传进程已在后台启动，PID: {pid}")
        print(f"日志文件: {log_file}")
        sys.exit(0)
    
    # 子进程继续执行上传任务
    try:
        upload_files_in_batches(file_list_path, config_dir, remote_dir)
        print("所有文件上传完成")
        logging.info("所有文件上传完成")
    except Exception as e:
        error_msg = f"上传过程中出现未预期的错误: {str(e)}"
        print(error_msg)
        logging.error(error_msg)