#!/usr/bin/env python3
import subprocess
import sys
import os

def upload_files(file_list_path, config_dir):
    """使用bypy上传文件列表中的文件，并指定配置目录"""
    # 读取文件列表
    with open(file_list_path, 'r') as f:
        # 跳过第一行（标题行）
        file_paths = f.readlines()[1:]
    
    # 为每个文件执行上传命令
    for file_path in file_paths:
        file_path = file_path.strip()
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        # 构造bypy命令，指定配置目录
        cmd = ['bypy', '--config-dir', config_dir, 'upload', file_path]
        
        try:
            print(f"正在上传: {file_path}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            print(f"上传成功: {file_path}")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"上传失败: {file_path}")
            print(f"错误信息: {e.stderr}")
        except subprocess.TimeoutExpired:
            print(f"上传超时: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python3 upload_all.py <file_list_path> <config_dir>")
        sys.exit(1)
    
    file_list_path = sys.argv[1]
    config_dir = sys.argv[2]
    
    upload_files(file_list_path, config_dir)