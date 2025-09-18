"""
基于并行处理框架的视频mask分析工作器
"""

import torch
import cv2
import numpy as np
import os
from collections import defaultdict
from typing import Dict, Any, List
from ultralytics import YOLO

from utils.parallel_processor import BaseGPUWorker


class MaskAnalysisWorker(BaseGPUWorker):
    """视频mask分析并行工作器"""
    
    def __init__(self, gpu_id: int, **kwargs):
        super().__init__(gpu_id, **kwargs)
        self.model = None
        self.batch_size = kwargs.get('batch_size', 8)
        
    def initialize_model(self, model_path: str, **kwargs):
        """初始化YOLO分割模型"""
        self.model = YOLO(model_path, verbose=False)
        self.model.to(self.device)
        self.model.model.eval()
        
        # 优化设置
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    def calculate_mask_change_ratio(self, prev_mask, curr_mask):
        """计算两个mask间变化点占总点数的比例"""
        if prev_mask is None or curr_mask is None:
            return 0.0
            
        # 计算变化的像素点
        changed_pixels = np.sum(prev_mask != curr_mask)
        
        # 计算当前mask的总像素点数
        total_pixels = np.sum(curr_mask > 0)
        
        if total_pixels == 0:
            return 0.0
            
        # 返回变化比例
        return changed_pixels / total_pixels
    
    def calculate_mask_changes(self, mask_sequence):
        """计算mask序列的变化统计"""
        if len(mask_sequence) < 2:
            return {'avg_change_ratio': 0.0}
            
        change_ratios = []
        
        for i in range(1, len(mask_sequence)):
            prev_mask = mask_sequence[i-1]
            curr_mask = mask_sequence[i]
            ratio = self.calculate_mask_change_ratio(prev_mask, curr_mask)
            change_ratios.append(ratio)
        
        return {
            'avg_change_ratio': np.mean(change_ratios) if change_ratios else 0.0,
            'max_change_ratio': np.max(change_ratios) if change_ratios else 0.0
        }
    
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """处理批次数据（基类要求的抽象方法）"""
        # 这个方法在当前实现中不直接使用，但需要实现以满足抽象基类要求
        results = []
        for item in batch:
            result = self.process_single(item)
            results.append(result)
        return results
    
    def process_video_frames_batch(self, video_path: str, frame_interval: int = 10):
        """批量处理视频帧，每frame_interval帧处理一次"""
        person_masks = defaultdict(list)
        frame_count = 0
        processed_count = 0
        
        # 使用YOLO track方法，更稳定且支持跟踪
        with torch.cuda.device(self.gpu_id):
            with torch.no_grad():
                for results in self.model.track(source=video_path, persist=True, stream=True, verbose=False):
                    frame_count += 1
                    
                    # 只在指定间隔的帧进行处理
                    if frame_count % frame_interval != 0:
                        continue
                        
                    processed_count += 1
                    
                    for result in results:
                        if result.masks is not None and result.boxes is not None:
                            masks = result.masks.data.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy().astype(int)
                            ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.arange(len(masks))
                            
                            # 只处理人类 (class 0)
                            for mask, cls, track_id in zip(masks, classes, ids):
                                if cls == 0:  # 人类
                                    mask_bin = (mask > 0.5).astype(np.uint8)
                                    person_masks[track_id].append(mask_bin)
        
        return person_masks, processed_count
    
    
    def process_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个视频文件"""
        # 兼容多种数据格式
        video_rel_path = data.get('video', data.get('path', ''))
        mhi_root = data.get('mhi_root', data.get('base_path', '/home/jinqiao/mhi'))
        frame_interval = data.get('frame_interval', 10)
        
        # 构建完整路径
        if os.path.isabs(video_rel_path):
            video_path = video_rel_path
        else:
            video_path = os.path.join(mhi_root, video_rel_path)
        
        if not os.path.exists(video_path):
            return {
                'main_persons_count': -1,
                'avg_person_area': -1,
                'avg_change_ratio': -1,
                'max_change_ratio': -1,
                'processed_frames': -1
            }
        
        try:
            # 批量处理视频帧
            person_masks, processed_count = self.process_video_frames_batch(video_path, frame_interval)
            
            if person_masks is None or processed_count == 0:
                return self._get_empty_result()
            
            # 找到主要人物
            main_persons = {}
            for track_id, mask_list in person_masks.items():
                if len(mask_list) >= 2:  # 至少出现2次
                    avg_area = np.mean([np.sum(mask) for mask in mask_list])
                    duration = len(mask_list)
                    score = avg_area * duration
                    main_persons[track_id] = {
                        'masks': mask_list,
                        'avg_area': avg_area,
                        'duration': duration,
                        'score': score
                    }
            
            if not main_persons:
                return self._get_empty_result()
            
            # 选择主要人物
            sorted_persons = sorted(main_persons.items(), key=lambda x: x[1]['score'], reverse=True)
            top_person = sorted_persons[0]  # 只取最主要的人物
            
            # 计算变化统计
            mask_sequence = top_person[1]['masks']
            changes = self.calculate_mask_changes(mask_sequence)
            
            return {
                'main_persons_count': 1,
                'avg_person_area': top_person[1]['avg_area'],
                'avg_change_ratio': changes['avg_change_ratio'],
                'max_change_ratio': changes['max_change_ratio'],
                'processed_frames': processed_count
            }
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return self._get_empty_result()
    
    def _get_empty_result(self):
        """返回空结果"""
        return {
            'main_persons_count': 0,
            'avg_person_area': 0,
            'avg_change_ratio': 0.0,
            'max_change_ratio': 0.0,
            'processed_frames': 0
        }