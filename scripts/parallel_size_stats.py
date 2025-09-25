#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
并行统计文件大小工具
用于统计CSV文件中列出的文件总大小
"""

import os
import csv
import argparse
import logging
import time
import sys
from multiprocessing import Pool, cpu_count
from functools import partial


def get_file_size(file_path):
    """获取单个文件的大小"""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        # 记录错误但不中断整个过程
        return 0


def process_batch(args):
    """处理一批文件路径，返回总大小和文件数"""
    file_paths, batch_id = args
    total_size = 0
    processed_count = 0
    error_count = 0
    
    for path in file_paths:
        try:
            size = get_file_size(path)
            total_size += size
            processed_count += 1
        except Exception as e:
            error_count += 1
    
    return total_size, processed_count, error_count, batch_id


def read_csv_paths(csv_file, path_column='path'):
    """从CSV文件中读取路径列"""
    paths = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        # 使用csv.reader处理可能包含逗号的路径
        reader = csv.reader(f)
        header = next(reader)  # 读取表头
        
        # 找到路径列的索引
        try:
            path_index = header.index(path_column)
        except ValueError:
            # 如果找不到指定的列名，默认使用第二列（索引1）
            path_index = 1
        
        for row in reader:
            if len(row) > path_index:
                paths.append(row[path_index])
    
    return paths


def main():
    parser = argparse.ArgumentParser(description='并行统计文件大小工具')
    parser.add_argument('--csv-file', '-f', required=True, help='CSV文件路径')
    parser.add_argument('--path-column', '-p', default='path', help='路径列名（默认:path）')
    parser.add_argument('--workers', '-w', type=int, default=0, help='工作进程数（默认为CPU核心数）')
    parser.add_argument('--batch-size', '-b', type=int, default=1000, help='每批处理的文件数（默认:1000）')
    parser.add_argument('--timeout', '-t', type=int, default=300, help='每个批次的超时时间（秒，默认:300）')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    logger.info(f"开始统计 {args.csv_file} 中文件的总大小")
    logger.info(f"使用 {args.workers if args.workers > 0 else max(1, cpu_count()//2)} 个工作进程")
    logger.info(f"批处理大小: {args.batch_size}")
    logger.info(f"超时时间: {args.timeout} 秒")
    
    # 读取CSV文件中的路径
    logger.info("正在读取CSV文件...")
    try:
        paths = read_csv_paths(args.csv_file, args.path_column)
        logger.info(f"读取到 {len(paths)} 个文件路径")
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        return 1
    
    if not paths:
        logger.warning("没有找到任何文件路径")
        return 0
    
    # 准备批处理数据
    batch_size = args.batch_size
    batches = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]
    logger.info(f"将数据分为 {len(batches)} 个批次处理")
    
    # 创建进程池
    workers = args.workers if args.workers > 0 else max(1, cpu_count()//2)
    pool = Pool(processes=workers)
    
    # 准备处理函数
    batch_tasks = [(batch, i) for i, batch in enumerate(batches)]
    
    # 并行处理
    logger.info("开始并行处理...")
    total_size = 0
    total_processed = 0
    total_errors = 0
    
    try:
        # 使用超时机制处理每个批次
        results = pool.map_async(process_batch, batch_tasks)
        
        # 等待结果，设置超时时间
        results = results.get(timeout=args.timeout * len(batches))
        
        # 收集结果
        for batch_size, processed_count, error_count, batch_id in results:
            total_size += batch_size
            total_processed += processed_count
            total_errors += error_count
            
            # 每处理10个批次显示一次进度
            if (batch_id + 1) % 10 == 0 or batch_id == len(batches) - 1:
                logger.info(f"已处理 {batch_id + 1}/{len(batches)} 个批次")
                logger.info(f"当前累计大小: {total_size} 字节")
        
        pool.close()
        pool.join()
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        pool.terminate()
        pool.join()
        return 1
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        pool.terminate()
        pool.join()
        return 1
    
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
    logger.info(f"总文件数: {len(paths)}")
    logger.info(f"成功处理: {total_processed}")
    logger.info(f"处理错误: {total_errors}")
    logger.info(f"总大小: {format_size(total_size)} ({total_size} 字节)")
    logger.info(f"耗时: {elapsed_time:.2f} 秒")
    logger.info(f"处理速度: {len(paths)/elapsed_time:.2f} 文件/秒")
    logger.info("=" * 50)
    
    # 将结果保存到文件
    result_file = args.csv_file.replace('.csv', '_size_stats.txt')
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"文件统计结果\n")
            f.write(f"=" * 30 + "\n")
            f.write(f"总文件数: {len(paths)}\n")
            f.write(f"成功处理: {total_processed}\n")
            f.write(f"处理错误: {total_errors}\n")
            f.write(f"总大小: {format_size(total_size)} ({total_size} 字节)\n")
            f.write(f"耗时: {elapsed_time:.2f} 秒\n")
            f.write(f"处理速度: {len(paths)/elapsed_time:.2f} 文件/秒\n")
        logger.info(f"结果已保存到 {result_file}")
    except Exception as e:
        logger.error(f"保存结果文件失败: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())