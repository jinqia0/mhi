#!/usr/bin/env python3
"""
批量视频Pose提取工具
处理videos目录中的所有视频文件并提取骨架数据

Usage:
    python batch_video_pose_extraction.py --input-dir videos --output-dir pose_results
"""

import os
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Queue
import threading
from typing import List, Dict, Any
import pandas as pd

# 导入我们的pose提取器
from video_pose_extraction import VideoPoseExtractor


class BatchPoseProcessor:
    def __init__(self, input_dir, output_dir, model_name='checkpoints/yolo11m-pose.pt', 
                 max_workers=4, save_visualizations=False, conf_threshold=0.5):
        """
        批量pose提取处理器
        
        Args:
            input_dir: 输入视频目录
            output_dir: 输出结果目录  
            model_name: YOLO pose模型路径
            max_workers: 最大并行worker数量
            save_visualizations: 是否保存可视化视频
            conf_threshold: 置信度阈值
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.max_workers = max_workers
        self.save_visualizations = save_visualizations
        self.conf_threshold = conf_threshold
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir = self.output_dir / "json_data"
        self.json_dir.mkdir(exist_ok=True)
        
        if self.save_visualizations:
            self.vis_dir = self.output_dir / "visualizations" 
            self.vis_dir.mkdir(exist_ok=True)
        
        # 进度跟踪
        self.progress_file = self.output_dir / "processing_progress.json"
        self.summary_file = self.output_dir / "processing_summary.csv"
        
        # 加载已处理的文件列表
        self.processed_files = self._load_progress()
        
    def _load_progress(self):
        """加载处理进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                return set(progress.get('processed_files', []))
        return set()
    
    def _save_progress(self, processed_files):
        """保存处理进度"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                'processed_files': list(processed_files),
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False)
    
    def find_video_files(self) -> List[Path]:
        """查找所有视频文件"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        video_files = []
        
        print(f"扫描目录: {self.input_dir}")
        for ext in video_extensions:
            files = list(self.input_dir.rglob(f"*{ext}"))
            video_files.extend(files)
            print(f"找到 {len(files)} 个 {ext} 文件")
        
        # 过滤已处理的文件
        remaining_files = [f for f in video_files if str(f) not in self.processed_files]
        
        print(f"总共找到: {len(video_files)} 个视频文件")
        print(f"已处理: {len(self.processed_files)} 个文件") 
        print(f"剩余待处理: {len(remaining_files)} 个文件")
        
        return remaining_files
    
    def process_single_video(self, video_path: Path) -> Dict[str, Any]:
        """
        处理单个视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            dict: 处理结果信息
        """
        result = {
            'video_path': str(video_path),
            'status': 'failed',
            'error': None,
            'processing_time': 0,
            'pose_count': 0,
            'frame_count': 0,
            'output_files': []
        }
        
        start_time = time.time()
        
        try:
            # 创建输出目录 (根据视频相对路径)
            rel_path = video_path.relative_to(self.input_dir)
            output_subdir = self.json_dir / rel_path.parent / rel_path.stem
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # 可视化输出目录
            vis_output_dir = None
            if self.save_visualizations:
                vis_output_dir = self.vis_dir / rel_path.parent / rel_path.stem
                vis_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"处理视频: {video_path}")
            
            # 创建pose提取器
            extractor = VideoPoseExtractor(
                model_name=self.model_name,
                device='auto',
                conf_threshold=self.conf_threshold
            )
            
            # 提取poses
            pose_data = extractor.extract_poses_from_video(
                video_path=video_path,
                output_dir=output_subdir,
                save_visualizations=self.save_visualizations
            )
            
            # 统计结果
            frame_count = len(pose_data['poses'])
            pose_count = sum(len(frame['detections']) for frame in pose_data['poses'])
            
            result.update({
                'status': 'success',
                'processing_time': time.time() - start_time,
                'pose_count': pose_count,
                'frame_count': frame_count,
                'output_files': [
                    str(output_subdir / f"{video_path.stem}_poses.json")
                ]
            })
            
            if self.save_visualizations:
                vis_file = vis_output_dir / f"{video_path.stem}_pose_vis.mp4"
                if vis_file.exists():
                    result['output_files'].append(str(vis_file))
            
            print(f"✓ 完成: {video_path} - {pose_count}个pose, {frame_count}帧, {result['processing_time']:.1f}s")
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            print(f"✗ 失败: {video_path} - {e}")
            
        return result
    
    def process_batch(self, video_files: List[Path]) -> List[Dict[str, Any]]:
        """
        批量处理视频文件
        
        Args:
            video_files: 视频文件列表
            
        Returns:
            List[Dict]: 处理结果列表
        """
        results = []
        processed_count = 0
        
        print(f"\n开始批量处理 {len(video_files)} 个视频文件...")
        print(f"并发worker数量: {self.max_workers}")
        
        # 使用线程池处理 (因为主要是IO密集型任务)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_video = {
                executor.submit(self.process_single_video, video_file): video_file 
                for video_file in video_files
            }
            
            # 收集结果
            for future in as_completed(future_to_video):
                video_file = future_to_video[future]
                try:
                    result = future.result()
                    results.append(result)
                    processed_count += 1
                    
                    # 更新进度
                    if result['status'] == 'success':
                        self.processed_files.add(result['video_path'])
                        
                        # 每处理10个文件保存一次进度
                        if processed_count % 10 == 0:
                            self._save_progress(self.processed_files)
                    
                    # 显示进度
                    progress = (processed_count / len(video_files)) * 100
                    print(f"进度: {processed_count}/{len(video_files)} ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"处理 {video_file} 时发生异常: {e}")
                    results.append({
                        'video_path': str(video_file),
                        'status': 'failed',
                        'error': str(e),
                        'processing_time': 0,
                        'pose_count': 0,
                        'frame_count': 0,
                        'output_files': []
                    })
        
        # 保存最终进度
        self._save_progress(self.processed_files)
        
        return results
    
    def save_summary(self, results: List[Dict[str, Any]]):
        """保存处理摘要"""
        df = pd.DataFrame(results)
        df.to_csv(self.summary_file, index=False, encoding='utf-8')
        
        # 计算统计信息
        successful = df[df['status'] == 'success']
        failed = df[df['status'] == 'failed']
        
        total_poses = successful['pose_count'].sum()
        total_frames = successful['frame_count'].sum()
        total_time = df['processing_time'].sum()
        
        print(f"\n=== 批量处理完成 ===")
        print(f"总文件数: {len(results)}")
        print(f"成功处理: {len(successful)} 个")
        print(f"处理失败: {len(failed)} 个")
        print(f"总pose数量: {total_poses}")
        print(f"总帧数: {total_frames}")
        print(f"总处理时间: {total_time:.1f}s ({total_time/3600:.1f}h)")
        print(f"平均每帧pose数: {total_poses/total_frames:.2f}" if total_frames > 0 else 0)
        print(f"处理速度: {len(successful)/total_time*3600:.1f} 文件/小时" if total_time > 0 else 0)
        print(f"结果摘要已保存到: {self.summary_file}")
        
        if len(failed) > 0:
            print(f"\n失败的文件:")
            for _, row in failed.iterrows():
                print(f"  {row['video_path']}: {row['error']}")
    
    def run(self):
        """运行批量处理"""
        print("=" * 60)
        print("YOLO Pose 批量提取工具")
        print("=" * 60)
        
        # 查找视频文件
        video_files = self.find_video_files()
        
        if not video_files:
            print("没有找到待处理的视频文件!")
            return
        
        # 批量处理
        results = self.process_batch(video_files)
        
        # 保存摘要
        self.save_summary(results)


def main():
    parser = argparse.ArgumentParser(description='批量视频Pose提取工具')
    parser.add_argument('--input-dir', '-i', default='videos', help='输入视频目录 (默认: videos)')
    parser.add_argument('--output-dir', '-o', default='pose_results', help='输出目录 (默认: pose_results)')
    parser.add_argument('--model', '-m', default='checkpoints/yolo11m-pose.pt', help='YOLO pose模型')
    parser.add_argument('--workers', '-w', type=int, default=4, help='并行worker数量 (默认: 4)')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--save-vis', action='store_true', help='保存可视化视频')
    parser.add_argument('--resume', action='store_true', help='从上次中断处继续处理')
    
    args = parser.parse_args()
    
    # 环境设置
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # 检查输入目录
    if not Path(args.input_dir).exists():
        print(f"错误: 输入目录不存在 - {args.input_dir}")
        return 1
    
    # 检查模型文件
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在 - {args.model}")
        return 1
    
    try:
        # 创建批量处理器
        processor = BatchPoseProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            max_workers=args.workers,
            save_visualizations=args.save_vis,
            conf_threshold=args.conf
        )
        
        # 运行批量处理
        processor.run()
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理，进度已保存。")
        print("使用 --resume 参数可以从中断处继续。")
        return 1
    except Exception as e:
        print(f"批量处理过程中出现错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())