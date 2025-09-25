#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分批统计文件大小工具
用于统计分割文件中列出的文件总大小
"""

import os
import glob
import argparse
import logging
import time
import sys


def get_file_size(file_path):
    """获取单个文件的大小"""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        return 0


def process_batch_file(batch_file):
    """处理单个批次文件"""
    total_size = 0
    file_count = 0
    error_count = 0
    
    with open(batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            path = line.strip()
            if path:
                try:
                    size = get_file_size(path)
                    total_size += size
                    file_count += 1
                except Exception as e:
                    error_count += 1
    
    return total_size, file_count, error_count


def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info("开始分批统计文件总大小")
    
    # 查找所有分割文件
    batch_files = sorted(glob.glob('split_files_*'))
    logger.info(f"找到 {len(batch_files)} 个批次文件")
    
    if not batch_files:
        logger.warning("没有找到任何批次文件")
        return 0
    
    # 逐个处理批次文件
    total_size = 0
    total_files = 0
    total_errors = 0
    
    for i, batch_file in enumerate(batch_files):
        logger.info(f"正在处理批次 {i+1}/{len(batch_files)}: {batch_file}")
        batch_start_time = time.time()
        
        batch_size, file_count, error_count = process_batch_file(batch_file)
        total_size += batch_size
        total_files += file_count
        total_errors += error_count
        
        batch_elapsed_time = time.time() - batch_start_time
        logger.info(f"批次 {i+1} 处理完成: 大小={batch_size} 字节, "
                   f"文件数={file_count}, 错误数={error_count}, "
                   f"耗时={batch_elapsed_time:.2f} 秒")
        
        # 每处理10个批次显示一次总体进度
        if (i + 1) % 10 == 0 or i == len(batch_files) - 1:
            progress = (i + 1) / len(batch_files) * 100
            elapsed_time = time.time() - start_time
            speed = (i + 1) / elapsed_time if elapsed_time > 0 else 0
            
            # 转换为人类可读的格式
            def format_size(size_bytes):
                if size_bytes == 0:
                    return "0B"
                size_names = ["B", "KB", "MB", "GB", "TB"]
                j = 0
                while size_bytes >= 1024.0 and j < len(size_names) - 1:
                    size_bytes /= 1024.0
                    j += 1
                return f"{size_bytes:.2f} {size_names[j]}"
            
            logger.info(f"总体进度: {progress:.1f}% ({i + 1}/{len(batch_files)}) "
                       f"已处理大小: {format_size(total_size)} "
                       f"处理速度: {speed:.2f} 批次/秒")
    
    # 计算最终结果
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 转换为人类可读的格式
    def format_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024.0 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.2f} {size_names[i]}"
    
    logger.info("=" * 50)
    logger.info("统计完成!")
    logger.info(f"总批次数: {len(batch_files)}")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"处理错误: {total_errors}")
    logger.info(f"总大小: {format_size(total_size)} ({total_size} 字节)")
    logger.info(f"耗时: {elapsed_time:.2f} 秒")
    logger.info(f"处理速度: {len(batch_files)/elapsed_time:.2f} 批次/秒")
    logger.info("=" * 50)
    
    # 将结果保存到文件
    try:
        with open('final_size_stats.txt', 'w', encoding='utf-8') as f:
            f.write(f"文件统计结果\n")
            f.write(f"=" * 30 + "\n")
            f.write(f"总批次数: {len(batch_files)}\n")
            f.write(f"总文件数: {total_files}\n")
            f.write(f"处理错误: {total_errors}\n")
            f.write(f"总大小: {format_size(total_size)} ({total_size} 字节)\n")
            f.write(f"耗时: {elapsed_time:.2f} 秒\n")
            f.write(f"处理速度: {len(batch_files)/elapsed_time:.2f} 批次/秒\n")
        logger.info("结果已保存到 final_size_stats.txt")
    except Exception as e:
        logger.error(f"保存结果文件失败: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())