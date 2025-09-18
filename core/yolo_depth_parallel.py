#!/usr/bin/env python3
"""
并行化深度接触判定脚本
基于 utils/parallel_processor.py 框架实现GPU并行加速
"""

import os
import sys
import warnings
import logging
import cv2
import pandas as pd
import numpy as np
import itertools
import contextlib
from typing import Dict, Any, List, Optional

# 设置基本环境变量
os.environ.update({
    'PYTHONWARNINGS': 'ignore',
    'TOKENIZERS_PARALLELISM': 'false',
})

# 抑制警告
warnings.filterwarnings('ignore')


# 导入torch和相关库
import torch
import torch.multiprocessing as mp

# 添加路径
sys.path.append('/home/jinqiao/mhi/third_party/Depth-Anything-V2')
sys.path.append('/home/jinqiao/mhi/utils')

from parallel_processor import BaseGPUWorker, ParallelVideoProcessor
from depth_anything_v2.dpt import DepthAnythingV2


class DepthContactWorker(BaseGPUWorker):
    """深度接触判定工作器"""
    
    def __init__(self, *args, **kwargs):
        """初始化工作器"""
        super().__init__(*args, **kwargs)
    
    def initialize_model(self, yolo_path: str, depth_encoder: str = 'vitb', **kwargs):
        """初始化YOLO和深度估计模型"""
        print(f"GPU {self.gpu_id}: 初始化模型...")
        
        # 初始化YOLO模型
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_path)
        
        # 初始化深度模型
        self.depth_model = self._init_depth_model(depth_encoder)
        
        # 算法参数
        self.contact_ratio_threshold = kwargs.get('contact_ratio_threshold', 0.1)
        self.similarity_threshold = kwargs.get('similarity_threshold', 0.01)
        self.area_ratio = kwargs.get('area_ratio', 0.5)
        self.min_area = kwargs.get('min_area', 1000)
        self.depth_input_size = kwargs.get('depth_input_size', 518)
        self.depth_batch_size = kwargs.get('depth_batch_size', 8)
        self.max_depth_frames = kwargs.get('max_depth_frames', 15)
        
        print(f"GPU {self.gpu_id}: 模型初始化完成")
    
    def _init_depth_model(self, encoder: str = 'vitb'):
        """初始化深度估计模型"""
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        depth_model = DepthAnythingV2(**model_configs[encoder])
        
        # 加载预训练权重
        checkpoint_path = f'/home/jinqiao/mhi/checkpoints/depth_anything_v2_{encoder}.pth'
        if os.path.exists(checkpoint_path):
            depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            raise FileNotFoundError(f"深度模型权重文件未找到: {checkpoint_path}")
        
        depth_model = depth_model.to(self.device).eval()
        
        # 启用性能优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        return depth_model
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理批次数据"""
        results = []
        for data in batch:
            result = self.process_single(data)
            results.append(result)
        return results
    
    def process_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个视频文件"""
        video_path = data.get('path', '')
        video_base_path = data.get('base_path', '/home/jinqiao/mhi')
        
        try:
            result = self._process_video_with_depth(video_path, video_base_path)
            return result
        except Exception as e:
            print(f"GPU {self.gpu_id}: 处理视频失败 {video_path}: {e}")
            return {
                'video': os.path.relpath(video_path, video_base_path),
                'contact_frames': -1,
                'depth_contact_frames': -1,
                'total_frames': -1,
                'contact_frame_ratio': -1.0,
                'depth_contact_frame_ratio': -1.0,
                'is_contact': -1,
                'is_depth_contact': -1,
                'error': str(e)
            }
    
    def _process_video_with_depth(self, video_path: str, base_path: str) -> Dict[str, Any]:
        """集成深度信息的视频接触判定"""
        frame_count = 0
        contact_frame_set = set()
        depth_contact_frame_set = set()
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        try:
            # YOLO跟踪
            yolo_results = self.yolo_model.track(
                source=video_path,
                persist=True,
                stream=True,
                verbose=False,
                save=False
            )
            
            contact_frames_candidates = []
            
            for results in yolo_results:
                frame_count += 1
                
                # 读取当前帧
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 提取人物masks
                masks_list, areas_list = self._extract_person_masks(results, frame.shape)
                
                if len(masks_list) < 2:
                    continue
                
                # 主要人物判定
                max_area = max(areas_list)
                main_person_indices = [
                    i for i, area in enumerate(areas_list) 
                    if area >= max_area * self.area_ratio and area >= self.min_area
                ]
                
                if len(main_person_indices) < 2:
                    continue
                
                # 基础接触判定
                contact_this_frame = self._check_basic_contact(masks_list, main_person_indices)
                if contact_this_frame:
                    contact_frame_set.add(frame_count)
                    contact_frames_candidates.append((frame_count, frame.copy(), masks_list, main_person_indices))
            
            # 深度检测
            if contact_frames_candidates:
                selected_frames = self._select_representative_frames(contact_frames_candidates)
                
                for batch_start in range(0, len(selected_frames), self.depth_batch_size):
                    batch_frames = selected_frames[batch_start:batch_start + self.depth_batch_size]
                    
                    frames_data = [(fcount, frame) for fcount, frame, _, _ in batch_frames]
                    depth_maps = self._batch_depth_inference(frames_data)
                    
                    for i, (fcount, _, masks_list, main_person_indices) in enumerate(batch_frames):
                        if i < len(depth_maps) and depth_maps[i] is not None:
                            depth_contact = self._check_depth_contact(masks_list, main_person_indices, depth_maps[i])
                            if depth_contact:
                                depth_contact_frame_set.add(fcount)
        
        finally:
            cap.release()
            torch.cuda.empty_cache()
        
        # 计算结果
        rel_path = os.path.relpath(video_path, base_path)
        contact_frames = len(contact_frame_set)
        depth_contact_frames = len(depth_contact_frame_set)
        
        contact_frame_ratio = contact_frames / frame_count if frame_count > 0 else 0
        depth_contact_frame_ratio = depth_contact_frames / frame_count if frame_count > 0 else 0
        
        is_contact = 1 if contact_frame_ratio >= self.contact_ratio_threshold else 0
        is_depth_contact = 1 if depth_contact_frame_ratio >= self.contact_ratio_threshold else 0
        
        return {
            'video': rel_path,
            'contact_frames': contact_frames,
            'depth_contact_frames': depth_contact_frames,
            'total_frames': frame_count,
            'contact_frame_ratio': f"{contact_frame_ratio:.3f}",
            'depth_contact_frame_ratio': f"{depth_contact_frame_ratio:.3f}",
            'is_contact': is_contact,
            'is_depth_contact': is_depth_contact
        }
    
    def _extract_person_masks(self, results, frame_shape):
        """提取人物masks"""
        masks_list = []
        areas_list = []
        
        frame_height, frame_width = frame_shape[:2]
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for mask, cls in zip(masks, classes):
                    if cls == 0:  # 人物类别
                        if mask.shape != (frame_height, frame_width):
                            mask_resized = cv2.resize(mask, (frame_width, frame_height))
                        else:
                            mask_resized = mask
                        
                        mask_bin = (mask_resized > 0.5).astype(np.uint8)
                        area = np.sum(mask_bin)
                        if area >= self.min_area:
                            masks_list.append(mask_bin)
                            areas_list.append(area)
        
        return masks_list, areas_list
    
    def _check_basic_contact(self, masks_list: List[np.ndarray], 
                           main_person_indices: List[int]) -> bool:
        """检查基础mask重叠"""
        for (i, mask1_idx), (j, mask2_idx) in itertools.combinations(enumerate(main_person_indices), 2):
            mask1 = masks_list[mask1_idx]
            mask2 = masks_list[mask2_idx]
            
            if np.any(np.logical_and(mask1, mask2)):
                return True
        return False
    
    def _check_depth_contact(self, masks_list: List[np.ndarray], 
                           main_person_indices: List[int], 
                           depth_map: np.ndarray) -> bool:
        """深度接触检查"""
        for (i, mask1_idx), (j, mask2_idx) in itertools.combinations(enumerate(main_person_indices), 2):
            mask1 = masks_list[mask1_idx]
            mask2 = masks_list[mask2_idx]
            
            is_similar, reason = self._is_depth_similar_at_contact(mask1, mask2, depth_map)
            if is_similar:
                return True
        return False
    
    def _batch_depth_inference(self, frames_data: List[tuple]) -> List[np.ndarray]:
        """批量深度推理"""
        if not frames_data:
            return []
        
        try:
            batch_frames = [frame for _, frame in frames_data]
            depth_maps = []
            
            with torch.no_grad():
                for frame in batch_frames:
                    try:
                        depth_map = self.depth_model.infer_image(frame, input_size=self.depth_input_size)
                        depth_maps.append(depth_map)
                    except Exception as e:
                        print(f"GPU {self.gpu_id}: 深度推理失败: {e}")
                        depth_maps.append(None)
            
            return depth_maps
            
        except Exception as e:
            print(f"GPU {self.gpu_id}: 批处理深度推理异常: {e}")
            return [None] * len(frames_data)
    
    def _select_representative_frames(self, contact_candidates: List[tuple]) -> List[tuple]:
        """选择代表性接触帧进行深度检测"""
        if len(contact_candidates) <= self.max_depth_frames:
            return contact_candidates
        
        # 均匀采样
        step = max(1, len(contact_candidates) // self.max_depth_frames)
        selected = []
        
        for i in range(0, len(contact_candidates), step):
            if len(selected) >= self.max_depth_frames:
                break
            selected.append(contact_candidates[i])
        
        # 添加关键位置帧
        if len(selected) < self.max_depth_frames:
            remaining = self.max_depth_frames - len(selected)
            selected_indices = set([c[0] for c in selected])
            
            key_positions = [
                len(contact_candidates) // 4,
                len(contact_candidates) // 2, 
                len(contact_candidates) * 3 // 4,
                len(contact_candidates) - 1
            ]
            
            for pos in key_positions:
                if remaining <= 0:
                    break
                if pos < len(contact_candidates) and contact_candidates[pos][0] not in selected_indices:
                    selected.append(contact_candidates[pos])
                    selected_indices.add(contact_candidates[pos][0])
                    remaining -= 1
        
        return selected[:self.max_depth_frames]
    
    def _is_depth_similar_at_contact(self, mask1: np.ndarray, mask2: np.ndarray, 
                                   depth_map: np.ndarray, similarity_threshold: float = None) -> tuple:
        """深度相似性检查"""
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        try:
            # 找到接触区域
            intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5)
            
            if not np.any(intersection):
                return False, "没有接触区域"
            
            contact_pixels = np.sum(intersection)
            
            # 扩展接触区域获得稳定深度估计
            kernel = np.ones((5, 5), np.uint8)
            expanded_contact = cv2.dilate(intersection.astype(np.uint8), kernel, iterations=1)
            
            # 获取两个人物的深度
            person1_region = np.logical_and(mask1 > 0.5, expanded_contact.astype(bool))
            person2_region = np.logical_and(mask2 > 0.5, expanded_contact.astype(bool))
            
            if not np.any(person1_region) or not np.any(person2_region):
                return False, "无法获取深度数据"
            
            # 计算深度值
            depth1_values = depth_map[person1_region]
            depth2_values = depth_map[person2_region]
            
            depth1_median = np.median(depth1_values)
            depth2_median = np.median(depth2_values)
            
            if depth1_median <= 0 or depth2_median <= 0:
                return False, f"深度数据异常"
            
            # 计算相对深度差异
            global_depth_range = np.max(depth_map) - np.min(depth_map)
            depth_diff = abs(depth1_median - depth2_median)
            
            if global_depth_range <= 0:
                return False, f"全局深度范围异常"
            
            relative_diff = depth_diff / global_depth_range
            is_similar = relative_diff < similarity_threshold
            
            reason = f"深度差异: {relative_diff:.3f}, 阈值: {similarity_threshold:.3f}, 接触像素: {contact_pixels}"
            return is_similar, reason
            
        except Exception as e:
            return False, f"深度检查异常: {str(e)}"


def process_existing_csv(input_csv: str, base_path: str, temp_csv: str):
    """处理现有CSV文件，提取视频路径"""
    df = pd.read_csv(input_csv)
    print(f"读取CSV: {input_csv}, 包含 {len(df)} 个视频")
    
    video_files = []
    for _, row in df.iterrows():
        video_rel_path = row['video']
        video_full_path = os.path.join(base_path, video_rel_path)
        
        if os.path.exists(video_full_path):
            video_files.append({
                'path': video_full_path,
                'base_path': base_path,
                'original_data': row.to_dict()
            })
        else:
            print(f"警告: 视频文件不存在 {video_full_path}")
    
    df_new = pd.DataFrame(video_files)
    df_new.to_csv(temp_csv, index=False)
    print(f"创建处理CSV: {temp_csv}, 包含 {len(video_files)} 个有效视频")
    
    return temp_csv


def merge_results_with_original(original_csv: str, processed_csv: str, output_csv: str):
    """合并原始数据和处理结果"""
    df_original = pd.read_csv(original_csv)
    df_processed = pd.read_csv(processed_csv)
    
    merged_data = []
    
    for _, row in df_processed.iterrows():
        try:
            import ast
            original_data = ast.literal_eval(row['original_data'])
        except:
            video_path = row['video']
            matching_rows = df_original[df_original['video'] == video_path]
            if len(matching_rows) > 0:
                original_data = matching_rows.iloc[0].to_dict()
            else:
                original_data = {}
        
        merged_row = original_data.copy()
        merged_row.update({
            'contact_frames': row['contact_frames'],
            'depth_contact_frames': row['depth_contact_frames'], 
            'total_frames': row['total_frames'],
            'contact_frame_ratio': row['contact_frame_ratio'],
            'depth_contact_frame_ratio': row['depth_contact_frame_ratio'],
            'is_contact': row['is_contact'],
            'is_depth_contact': row['is_depth_contact']
        })
        
        merged_data.append(merged_row)
    
    df_merged = pd.DataFrame(merged_data)
    df_merged.to_csv(output_csv, index=False)
    print(f"合并结果保存到: {output_csv}")


def main():
    """主函数"""
    # 配置参数
    config = {
        'yolo_path': '/home/jinqiao/mhi/checkpoints/yolo11n-seg.pt',
        'input_csv': '/home/jinqiao/mhi/results/contact_stats_main_mask_analysis.csv',
        'base_path': '/home/jinqiao/mhi',
        'output_csv': '/home/jinqiao/mhi/results/contact_stats_depth_enhanced.csv',
        'temp_csv': 'video_list_depth_temp.csv',
        'depth_encoder': 'vits',
        
        # 并行化参数
        'num_gpus': 4,
        'processes_per_gpu': 2,
        'batch_size': 8,
        'chunk_size': 100,
        
        # 算法参数
        'contact_ratio_threshold': 0.1,
        'similarity_threshold': 0.01,
        'area_ratio': 0.5,
        'min_area': 1000,
        'depth_input_size': 518,
        'depth_batch_size': 8,
        'max_depth_frames': 15
    }
    
    print("=== 并行化深度接触判定 ===")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"输入CSV: {config['input_csv']}")
    print(f"深度模型: {config['depth_encoder']}")
    
    # 处理CSV文件
    process_existing_csv(
        config['input_csv'], 
        config['base_path'], 
        config['temp_csv']
    )
    
    # 创建并行处理器
    try:
        processor = ParallelVideoProcessor(
            worker_class=DepthContactWorker,
            num_gpus=config['num_gpus'],
            processes_per_gpu=config['processes_per_gpu'],
            batch_size=config['batch_size'],
            chunk_size=config['chunk_size'],
            temp_dir='temp_depth_processing',
            keep_temp=False
        )
    except Exception as e:
        print(f"处理器初始化失败: {e}")
        # 回退到单进程模式
        config['processes_per_gpu'] = 1
        config['num_gpus'] = 1
        processor = ParallelVideoProcessor(
            worker_class=DepthContactWorker,
            num_gpus=config['num_gpus'],
            processes_per_gpu=config['processes_per_gpu'],
            batch_size=config['batch_size'],
            chunk_size=config['chunk_size'],
            temp_dir='temp_depth_processing',
            keep_temp=False
        )
    
    # 开始处理
    print("开始并行处理...")
    try:
        output_path = processor.process_csv(
            csv_path=config['temp_csv'],
            output_path=config['output_csv'],
            columns=['path', 'base_path', 'original_data'],
            **{k: v for k, v in config.items() if k not in ['temp_csv', 'input_csv', 'base_path', 'output_csv']}
        )
        
        # 合并结果
        merge_results_with_original(config['input_csv'], output_path, config['output_csv'])
        
        # 统计结果
        df_result = pd.read_csv(config['output_csv'])
        df_success = df_result[df_result['total_frames'] > 0]
        total_videos = len(df_success)
        
        if total_videos > 0:
            contact_videos = len(df_success[df_success['is_contact'] == 1])
            depth_contact_videos = len(df_success[df_success['is_depth_contact'] == 1])
            
            contact_ratio = contact_videos / total_videos
            depth_contact_ratio = depth_contact_videos / total_videos
            
            print(f"\n=== 处理完成 ===")
            print(f"成功处理视频: {total_videos}")
            print(f"结果保存到: {config['output_csv']}")
            print(f"原始接触视频: {contact_videos} / {total_videos} ({contact_ratio:.3f})")
            print(f"深度接触视频: {depth_contact_videos} / {total_videos} ({depth_contact_ratio:.3f})")
            print(f"过滤效果: 过滤掉 {contact_videos - depth_contact_videos} 个误检视频")
            
            df_error = df_result[df_result['total_frames'] < 0]
            if len(df_error) > 0:
                print(f"处理失败视频: {len(df_error)}")
        else:
            print("没有成功处理的视频")
    
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        raise
    
    finally:
        if os.path.exists(config['temp_csv']):
            os.remove(config['temp_csv'])
            print("清理临时文件完成")


if __name__ == "__main__":
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()