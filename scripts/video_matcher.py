#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频文件匹配工具
用于从大规模视频集中根据CSV文件中的文件名筛选子集
"""

import os
import csv
import json
import argparse
import logging
import signal
import sys
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time
import glob


class VideoMatcher:
    def __init__(self, config):
        self.config = config
        self.index = {}  # 文件名到路径的映射
        self.matched_files = {}  # 已匹配的文件
        self.unmatched_files = set()  # 未匹配的文件名
        self.checkpoint_file = config.get('checkpoint_file', 'checkpoint.json')
        self.output_file = config.get('output_file', 'matched_paths.csv')
        self.progress_file = config.get('progress_file', 'progress.log')
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.progress_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 中断信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """处理中断信号，保存检查点"""
        self.logger.info(f"接收到信号 {signum}，正在保存检查点...")
        self._save_checkpoint()
        self.logger.info("检查点已保存，程序退出。")
        sys.exit(0)
        
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'index': self.index,  # 保存实际的索引数据
            'matched_files': self.matched_files,
            'unmatched_files': list(self.unmatched_files),  # set需要转换为list才能序列化
            'timestamp': time.time()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            
    def _load_checkpoint(self):
        """加载检查点"""
        if not os.path.exists(self.checkpoint_file):
            return False
            
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                
            # 加载索引数据
            self.index = checkpoint.get('index', {})
            self.matched_files = checkpoint.get('matched_files', {})
            
            # 转换unmatched_files回set类型
            unmatched_list = checkpoint.get('unmatched_files', [])
            self.unmatched_files = set(unmatched_list)
            
            self.logger.info(f"检查点加载成功，索引包含 {len(self.index)} 个文件，已匹配 {len(self.matched_files)} 个文件，未匹配 {len(self.unmatched_files)} 个文件")
            return True
        except Exception as e:
            self.logger.error(f"检查点加载失败: {e}")
            return False
            
    def _scan_directory(self, directory):
        """扫描单个目录，返回文件名到路径的映射"""
        file_map = {}
        file_count = 0
        try:
            # 使用os.walk而非glob，避免递归深度过大导致的性能问题
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith('.mp4'):
                        file_path = os.path.join(root, file)
                        # 避免重复文件名覆盖
                        if file not in file_map:
                            file_map[file] = file_path
                            file_count += 1
                            
                # 每处理10000个文件就记录一次进度
                if file_count % 10000 == 0:
                    self.logger.info(f"目录 {directory} 已扫描 {file_count} 个文件")
                    
            self.logger.info(f"目录 {directory} 扫描完成，共 {file_count} 个文件")
            return file_map
        except Exception as e:
            self.logger.error(f"扫描目录 {directory} 时出错: {e}")
            return {}
            
    def build_index(self, directories):
        """构建文件名索引"""
        self.logger.info("开始构建文件名索引...")
        start_time = time.time()
        
        # 检查是否已加载检查点
        checkpoint_loaded = self._load_checkpoint()
        if checkpoint_loaded:
            self.logger.info(f"从检查点恢复，索引已包含 {len(self.index)} 个文件")
                
        # 展开通配符路径
        expanded_directories = []
        for directory in directories:
            if '*' in directory:
                expanded_directories.extend(glob.glob(directory))
            else:
                expanded_directories.append(directory)
                
        self.logger.info(f"展开后共有 {len(expanded_directories)} 个目录需要扫描")
        
        # 如果从检查点恢复，跳过已经处理过的目录
        # 这里我们简化处理，如果已加载检查点且索引不为空，则跳过索引构建
        if checkpoint_loaded and len(self.index) > 0:
            self.logger.info("从检查点恢复索引，跳过索引构建阶段")
        else:
            # 逐个处理目录以避免内存溢出
            for i, directory in enumerate(expanded_directories):
                self.logger.info(f"正在处理目录 {i+1}/{len(expanded_directories)}: {directory}")
                
                # 扫描单个目录
                dir_index = self._scan_directory(directory)
                
                # 合并结果
                self.index.update(dir_index)
                
                # 每处理1个目录就保存检查点
                self._save_checkpoint()
                self.logger.info(f"已处理目录 {i+1}，当前索引包含 {len(self.index)} 个文件")
                
            end_time = time.time()
            self.logger.info(f"索引构建完成，共 {len(self.index)} 个文件，耗时 {end_time - start_time:.2f} 秒")
        
        # 保存最终检查点
        self._save_checkpoint()
        
    def _extract_filename(self, path):
        """从路径中提取文件名"""
        return os.path.basename(path)
        
    def load_target_files(self, csv_file, path_column):
        """加载目标文件名"""
        target_files = set()
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                total_rows = 0
                for row in reader:
                    filename = self._extract_filename(row[path_column])
                    target_files.add(filename)
                    total_rows += 1
                    
                    # 每处理10万行就记录一次进度
                    if total_rows % 100000 == 0:
                        self.logger.info(f"已加载 {total_rows} 行数据")
                        
            self.unmatched_files = target_files
            self.logger.info(f"加载了 {len(target_files)} 个目标文件名 (共处理 {total_rows} 行数据)")
            return True
        except Exception as e:
            self.logger.error(f"加载目标文件失败: {e}")
            return False
            
    def match_files(self):
        """匹配文件路径"""
        self.logger.info("开始匹配文件路径...")
        start_time = time.time()
        
        # 从检查点恢复已匹配的文件
        matched_count = len(self.matched_files)
        initial_unmatched_count = len(self.unmatched_files)
        
        self.logger.info(f"从检查点恢复，已匹配 {matched_count} 个文件，未匹配 {initial_unmatched_count} 个文件")
        
        # 如果没有未匹配的文件，直接返回
        if initial_unmatched_count == 0:
            self.logger.info("没有需要匹配的文件")
            return
            
        # 分批处理以避免内存问题
        unmatched_list = list(self.unmatched_files)  # 转换为列表以保持迭代稳定性
        batch_size = 100000  # 每批处理的文件数
        total_batches = (len(unmatched_list) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(unmatched_list))
            batch_files = unmatched_list[batch_start:batch_end]
            
            self.logger.info(f"正在处理批次 {batch_idx+1}/{total_batches}，包含 {len(batch_files)} 个文件")
            
            # 处理批次中的文件
            batch_progress_interval = max(1, len(batch_files) // 10)
            for i, filename in enumerate(batch_files):
                if filename in self.index:
                    self.matched_files[filename] = self.index[filename]
                    self.unmatched_files.discard(filename)  # 使用discard避免KeyError
                    matched_count += 1
                    
                # 批次内进度显示
                if (i + 1) % batch_progress_interval == 0 or i == len(batch_files) - 1:
                    progress = (i + 1) / len(batch_files) * 100
                    self.logger.info(f"批次 {batch_idx+1} 进度: {progress:.1f}% "
                                   f"已匹配: {matched_count}, 未匹配: {len(self.unmatched_files)}")
                    
            # 每处理完一个批次就保存检查点
            self._save_checkpoint()
            self.logger.info(f"批次 {batch_idx+1} 处理完成，已保存检查点")
            
        end_time = time.time()
        self.logger.info(f"匹配完成，耗时 {end_time - start_time:.2f} 秒")
        self.logger.info(f"总计: 已匹配 {len(self.matched_files)} 个文件, "
                       f"未匹配 {len(self.unmatched_files)} 个文件")
                       
        # 保存最终检查点
        self._save_checkpoint()
        
    def save_results(self):
        """保存匹配结果"""
        try:
            with open(self.output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'path'])  # 写入表头
                
                for filename, path in self.matched_files.items():
                    writer.writerow([filename, path])
                    
            self.logger.info(f"匹配结果已保存到 {self.output_file}")
            
            # 保存未匹配的文件名
            unmatched_file = self.output_file.replace('.csv', '_unmatched.txt')
            with open(unmatched_file, 'w', encoding='utf-8') as f:
                for filename in self.unmatched_files:
                    f.write(filename + '\n')
                    
            self.logger.info(f"未匹配文件名已保存到 {unmatched_file}")
            return True
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            return False
            
    def run(self):
        """运行主程序"""
        # 构建索引
        self.build_index(self.config['directories'])
        
        # 加载目标文件
        # 如果从检查点恢复，可能已经加载了部分目标文件信息
        if len(self.unmatched_files) == 0:
            if not self.load_target_files(self.config['csv_file'], self.config['path_column']):
                return False
        else:
            self.logger.info(f"从检查点恢复 {len(self.unmatched_files)} 个未匹配文件")
            
        # 匹配文件
        self.match_files()
        
        # 保存结果
        self.save_results()
        
        return True


def main():
    parser = argparse.ArgumentParser(description='视频文件匹配工具')
    parser.add_argument('--config', '-c', required=True, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置文件
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return 1
        
    # 运行匹配器
    matcher = VideoMatcher(config)
    if matcher.run():
        print("处理完成!")
        return 0
    else:
        print("处理失败!")
        return 1


if __name__ == '__main__':
    sys.exit(main())