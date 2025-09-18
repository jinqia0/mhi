#!/usr/bin/env python3
"""
并行Mask变化分析CSV处理脚本
使用方法: python run_mask_analysis.py [input_csv] [output_csv] [--frame-interval N] [--max-videos N]
"""

import sys
import os
import pandas as pd
from tqdm import tqdm
import argparse
import torch

# 导入并行处理相关模块
sys.path.append('/home/jinqiao/mhi')
from utils.parallel_processor import ParallelVideoProcessor
from core.mask_analysis_worker import MaskAnalysisWorker

def main():
    parser = argparse.ArgumentParser(description='Process videos with parallel mask change analysis')
    parser.add_argument('input_csv', nargs='?', default='/home/jinqiao/mhi/results/contact_stats_depth.csv',
                       help='Input CSV file path')
    parser.add_argument('output_csv', nargs='?', default='/home/jinqiao/mhi/results/contact_stats_depth_mask.csv',
                       help='Output CSV file path')
    parser.add_argument('--frame-interval', type=int, default=10,
                       help='Process every N frames (default: 10)')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='Maximum number of videos to process')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start processing from row N (default: 0)')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='Chunk size for parallel processing (default: 100)')
    parser.add_argument('--pool-size-per-gpu', type=int, default=2,
                       help='Number of workers per GPU (default: 2)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                       help='Comma-separated list of GPU IDs to use (default: 0,1,2,3)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV file {args.input_csv} does not exist")
        return
    
    print(f"Loading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Total videos in CSV: {len(df)}")
    
    # 处理范围
    start_idx = args.start_from
    if args.max_videos:
        end_idx = min(start_idx + args.max_videos, len(df))
    else:
        end_idx = len(df)
    
    print(f"Processing videos {start_idx} to {end_idx-1} (total: {end_idx - start_idx})")
    
    # 解析GPU列表
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    available_gpus = list(range(torch.cuda.device_count()))
    gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id in available_gpus]
    
    if not gpu_ids:
        print("No available GPUs found, using CPU")
        gpu_ids = [0]  # 使用默认设备
    
    print(f"Using GPUs: {gpu_ids}")
    print(f"Workers per GPU: {args.pool_size_per_gpu}")
    
    # 添加新列
    if 'avg_change_ratio' not in df.columns:
        df['avg_change_ratio'] = -1.0
    if 'max_change_ratio' not in df.columns:
        df['max_change_ratio'] = -1.0
    if 'processed_frames' not in df.columns:
        df['processed_frames'] = -1
    
    # 筛选需要处理的数据
    df_subset = df.iloc[start_idx:end_idx].copy()
    
    # 只处理未完成的视频
    mask_to_process = df_subset['avg_change_ratio'] < 0
    df_to_process = df_subset[mask_to_process].copy()
    
    if len(df_to_process) == 0:
        print("All videos have already been processed!")
        return
    
    print(f"Videos to process: {len(df_to_process)}")
    
    # 准备数据格式 - 添加frame_interval参数
    data_items = []
    for idx, row in df_to_process.iterrows():
        data_items.append({
            'video': row['video'],
            'mhi_root': '/home/jinqiao/mhi',
            'frame_interval': args.frame_interval,
            'original_index': idx
        })
    
    # 创建并行处理器
    processor = ParallelVideoProcessor(
        worker_class=MaskAnalysisWorker,
        num_gpus=len(gpu_ids),
        processes_per_gpu=args.pool_size_per_gpu,
        batch_size=8,
        chunk_size=args.chunk_size,
        temp_dir='temp_mask_processing',
        keep_temp=False
    )
    
    print("Starting parallel processing...")
    
    try:
        # 创建临时CSV文件用于处理
        temp_csv_path = '/tmp/mask_analysis_temp.csv'
        temp_df = pd.DataFrame(data_items)
        temp_df['path'] = temp_df['video']  # ParallelVideoProcessor 期望 'path' 列
        temp_df['base_path'] = temp_df['mhi_root']
        temp_df.to_csv(temp_csv_path, index=False)
        
        # 执行并行处理
        output_path = processor.process_csv(
            csv_path=temp_csv_path,
            output_path=None,
            columns=['path', 'base_path', 'frame_interval', 'original_index'],
            model_path='/home/jinqiao/mhi/checkpoints/yolo11n-seg.pt',
            batch_size=8
        )
        
        print(f"Parallel processing completed!")
        
        # 读取处理结果
        results_df = pd.read_csv(output_path)
        
        # 更新原始数据框
        for _, result_row in results_df.iterrows():
            if 'original_index' in result_row:
                original_idx = int(result_row['original_index'])
                
                # 更新结果到原始数据框
                df.loc[original_idx, 'avg_change_ratio'] = result_row.get('avg_change_ratio', -1)
                df.loc[original_idx, 'max_change_ratio'] = result_row.get('max_change_ratio', -1)
                df.loc[original_idx, 'processed_frames'] = result_row.get('processed_frames', -1)
        
        # 清理临时文件
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # 保存结果
        df.to_csv(args.output_csv, index=False)
        print(f"Results saved to: {args.output_csv}")
        
        # 打印统计信息
        valid_results = df[df['avg_change_ratio'] >= 0]
        if len(valid_results) > 0:
            print(f"\nMask Change Statistics:")
            print(f"  Total processed videos: {len(valid_results)}")
            print(f"  Average change ratio: {valid_results['avg_change_ratio'].mean():.4f} ± {valid_results['avg_change_ratio'].std():.4f}")
            print(f"  Max change ratio: {valid_results['max_change_ratio'].mean():.4f} ± {valid_results['max_change_ratio'].std():.4f}")
            print(f"  High motion videos (avg_change_ratio > 0.1): {len(valid_results[valid_results['avg_change_ratio'] > 0.1])}")
            print(f"  Very high motion videos (avg_change_ratio > 0.2): {len(valid_results[valid_results['avg_change_ratio'] > 0.2])}")
            
            # 显示变化比例分布
            print(f"\nChange Ratio Distribution:")
            bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0]
            labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30%+']
            
            for i in range(len(bins)-1):
                count = len(valid_results[
                    (valid_results['avg_change_ratio'] >= bins[i]) & 
                    (valid_results['avg_change_ratio'] < bins[i+1])
                ])
                percentage = count / len(valid_results) * 100
                print(f"  {labels[i]}: {count} videos ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'processor' in locals() and hasattr(processor, 'data_processor'):
            processor.data_processor.cleanup()

if __name__ == '__main__':
    main()