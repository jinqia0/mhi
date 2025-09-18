import os
import sys
import contextlib
import itertools
import numpy as np
import torch
import pandas as pd
from typing import Dict, Any
from ultralytics import YOLO

# 添加utils路径以导入并行处理框架
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from parallel_processor import BaseGPUWorker, ParallelVideoProcessor


class ContactDetectionWorker(BaseGPUWorker):
    """人际接触检测工作器"""
    
    def initialize_model(self, model_path: str, mhi_root: str = '/home/jinqiao/mhi', **kwargs):
        """初始化YOLO模型和参数"""
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.model = YOLO(model_path, verbose=False)
        self.model.to(self.device)
        self.model.model.eval()
        
        self.mhi_root = mhi_root
        self.contact_ratio_threshold = kwargs.get('contact_ratio_threshold', 0.1)
        
        # COCO数据集中的可交互物体类别
        self.interactive_objects = {32, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 66, 67, 73}
    
    def process_batch(self, batch: list) -> list:
        """处理批次数据（继承自基类的抽象方法）"""
        return [self.process_single(item) for item in batch]
    
    def process_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个视频文件"""
        video_path = data.get('video', '')
        
        # 构建完整视频路径 - 支持绝对路径
        if os.path.isabs(video_path):
            full_video_path = video_path
        else:
            # 尝试不同的路径组合
            full_video_path = os.path.join(self.mhi_root, video_path)
            if not os.path.exists(full_video_path):
                full_video_path = video_path  # 如果是相对于项目根目录的路径
            
        if not os.path.exists(full_video_path):
            print(f"视频文件不存在: {full_video_path}")
            return self._get_default_result_with_existing_data(data)
        
        try:
            result = self._process_video_file(video_path, full_video_path)
            # 保留所有现有数据，只添加新的间接接触检测结果
            return self._merge_with_existing_data(data, result)
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {e}")
            return self._get_default_result_with_existing_data(data)
    
    def _get_default_result_with_existing_data(self, existing_data: Dict[str, Any]) -> Dict[str, Any]:
        """返回包含现有数据的默认结果"""
        result = existing_data.copy()
        # 只添加新的间接接触检测列
        result.update({
            'indirect_contact_frames': 0,
            'indirect_contact_ratio': '0.000',
            'is_indirect_contact': 0,
            'total_enhanced_contact_frames': existing_data.get('contact_frames', 0),
            'total_enhanced_contact_ratio': existing_data.get('contact_frame_ratio', '0.000'),
            'is_any_enhanced_contact': existing_data.get('is_contact', 0)
        })
        return result
    
    def _merge_with_existing_data(self, existing_data: Dict[str, Any], new_result: Dict[str, Any]) -> Dict[str, Any]:
        """合并现有数据和新的检测结果"""
        result = existing_data.copy()
        
        # 从新结果中提取间接接触数据
        indirect_frames = new_result.get('indirect_contact_frames', 0)
        indirect_ratio = new_result.get('indirect_contact_ratio', '0.000')
        is_indirect = new_result.get('is_indirect_contact', 0)
        
        # 计算增强的总接触（原有直接接触 + 新的间接接触）
        existing_contact_frames = existing_data.get('contact_frames', 0)
        total_frames = existing_data.get('total_frames', 1)
        
        # 合并直接和间接接触帧（去重）
        total_enhanced_frames = existing_contact_frames + indirect_frames  # 简化处理，假设不重叠
        total_enhanced_ratio = total_enhanced_frames / total_frames if total_frames > 0 else 0
        is_any_enhanced = 1 if (existing_data.get('is_contact', 0) == 1 or is_indirect == 1) else 0
        
        # 添加新的列
        result.update({
            'indirect_contact_frames': indirect_frames,
            'indirect_contact_ratio': indirect_ratio,
            'is_indirect_contact': is_indirect,
            'total_enhanced_contact_frames': total_enhanced_frames,
            'total_enhanced_contact_ratio': f"{total_enhanced_ratio:.3f}",
            'is_any_enhanced_contact': is_any_enhanced
        })
        
        return result
    
    def _process_video_file(self, rel_video_path: str, full_video_path: str) -> Dict[str, Any]:
        """具体的视频处理实现"""
        frame_count = 0
        direct_contact_frame_set = set()
        indirect_contact_frame_set = set()
        
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            for results in self.model.track(
                source=full_video_path, 
                persist=True, 
                stream=True, 
                verbose=False,
                save=False  # 不保存跟踪结果以节省空间
            ):
                frame_count += 1
                human_masks_list = []
                human_areas_list = []
                object_masks_list = []
                
                for result in results:
                    if result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for mask, cls in zip(masks, classes):
                            mask_bin = (mask > 0.5).astype(np.uint8)
                            if cls == 0:  # 人类
                                area = np.sum(mask_bin)
                                human_masks_list.append(mask_bin)
                                human_areas_list.append(area)
                            elif cls in self.interactive_objects:  # 交互物体
                                object_masks_list.append(mask_bin)
                
                # 主要人物判定和接触检测
                if human_areas_list:
                    max_area = max(human_areas_list)
                    main_person_indices = [i for i, area in enumerate(human_areas_list) 
                                         if area >= max_area * 0.5]
                    
                    # 1. 检测直接的人-人接触
                    direct_contact_this_frame = False
                    for i, j in itertools.combinations(main_person_indices, 2):
                        intersection = np.logical_and(human_masks_list[i], human_masks_list[j])
                        if np.sum(intersection) > 0:
                            direct_contact_this_frame = True
                            break
                    
                    # 2. 检测通过物体的间接接触 (人-物-人)
                    indirect_contact_this_frame = False
                    if len(main_person_indices) >= 2 and object_masks_list:
                        for obj_mask in object_masks_list:
                            interacting_humans = []
                            for person_idx in main_person_indices:
                                human_mask = human_masks_list[person_idx]
                                human_object_intersection = np.logical_and(human_mask, obj_mask)
                                if np.sum(human_object_intersection) > 0:
                                    interacting_humans.append(person_idx)
                            
                            if len(interacting_humans) >= 2:
                                indirect_contact_this_frame = True
                                break
                    
                    if direct_contact_this_frame:
                        direct_contact_frame_set.add(frame_count)
                    if indirect_contact_this_frame:
                        indirect_contact_frame_set.add(frame_count)
        
        # 计算统计结果
        direct_contact_frames = len(direct_contact_frame_set)
        indirect_contact_frames = len(indirect_contact_frame_set)
        total_contact_frames = len(direct_contact_frame_set | indirect_contact_frame_set)
        
        direct_contact_ratio = direct_contact_frames / frame_count if frame_count > 0 else 0
        indirect_contact_ratio = indirect_contact_frames / frame_count if frame_count > 0 else 0
        total_contact_ratio = total_contact_frames / frame_count if frame_count > 0 else 0
        
        return {
            'direct_contact_frames': direct_contact_frames,
            'indirect_contact_frames': indirect_contact_frames,
            'total_contact_frames': total_contact_frames,
            'total_frames': frame_count,
            'direct_contact_ratio': f"{direct_contact_ratio:.3f}",
            'indirect_contact_ratio': f"{indirect_contact_ratio:.3f}",
            'total_contact_ratio': f"{total_contact_ratio:.3f}",
            'is_direct_contact': 1 if direct_contact_ratio >= self.contact_ratio_threshold else 0,
            'is_indirect_contact': 1 if indirect_contact_ratio >= self.contact_ratio_threshold else 0,
            'is_any_contact': 1 if total_contact_ratio >= self.contact_ratio_threshold else 0
        }

if __name__ == '__main__':
    # 配置参数
    model_path = '/home/jinqiao/mhi/checkpoints/yolo11n-seg.pt'
    mhi_root = '/home/jinqiao/mhi'
    input_csv = 'results/contact_stats_depth_mask.csv'
    output_csv = 'results/contact_stats_depth_mask_enhanced.csv'
    
    # 检查输入文件是否存在
    input_csv_path = os.path.join(mhi_root, input_csv)
    if not os.path.exists(input_csv_path):
        print(f"输入CSV文件不存在: {input_csv_path}")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    
    # 创建并行处理器
    processor = ParallelVideoProcessor(
        worker_class=ContactDetectionWorker,
        num_gpus=num_gpus,
        processes_per_gpu=8,  # 视频处理占用GPU内存较多，每个GPU只用1个进程
        batch_size=1,         # 每次处理1个视频
        chunk_size=100,         # 每个分块5个视频进行测试
        keep_temp=False
    )
    
    print(f"开始处理CSV文件: {input_csv}")
    print("使用并行处理框架进行人际接触检测...")
    
    # 处理CSV文件
    try:
        output_path = processor.process_csv(
            csv_path=input_csv_path,
            output_path=os.path.join(mhi_root, output_csv),
            columns=None,  # 读取所有列以保留现有数据
            model_path=model_path,
            mhi_root=mhi_root,
            contact_ratio_threshold=0.1
        )
        
        print(f"处理完成！结果保存到: {output_path}")
        
        # 读取结果并统计
        result_df = pd.read_csv(output_path)
        total_videos = len(result_df)
        
        if total_videos > 0:
            # 原有接触统计
            original_contact_videos = result_df['is_contact'].sum()
            depth_contact_videos = result_df['is_depth_contact'].sum()
            
            # 新增的间接接触统计
            indirect_contact_videos = result_df['is_indirect_contact'].sum() 
            enhanced_contact_videos = result_df['is_any_enhanced_contact'].sum()
            
            original_ratio = original_contact_videos / total_videos
            depth_ratio = depth_contact_videos / total_videos
            indirect_ratio = indirect_contact_videos / total_videos
            enhanced_ratio = enhanced_contact_videos / total_videos
            
            print(f"\n=== 处理结果统计 ===")
            print(f"总视频数量: {total_videos}")
            print(f"原有直接接触视频数量: {original_contact_videos} ({original_ratio:.3f})")
            print(f"深度接触视频数量: {depth_contact_videos} ({depth_ratio:.3f})")
            print(f"新增间接接触视频数量: {indirect_contact_videos} ({indirect_ratio:.3f})")
            print(f"增强后任意接触视频数量: {enhanced_contact_videos} ({enhanced_ratio:.3f})")
            
            # 计算平均统计
            avg_total_frames = result_df['total_frames'].mean()
            avg_original_ratio = result_df['contact_frame_ratio'].mean()
            avg_depth_ratio = result_df['depth_contact_frame_ratio'].mean()
            avg_indirect_ratio = result_df['indirect_contact_ratio'].astype(float).mean()
            avg_enhanced_ratio = result_df['total_enhanced_contact_ratio'].astype(float).mean()
            
            print(f"\n=== 平均统计信息 ===")
            print(f"平均视频帧数: {avg_total_frames:.1f}")
            print(f"平均原有接触比率: {avg_original_ratio:.3f}")
            print(f"平均深度接触比率: {avg_depth_ratio:.3f}")
            print(f"平均间接接触比率: {avg_indirect_ratio:.3f}")
            print(f"平均增强后接触比率: {avg_enhanced_ratio:.3f}")
        else:
            print("没有处理任何视频")
            
    except Exception as e:
        print(f"处理过程中出错: {e}")
        sys.exit(1)