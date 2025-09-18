import cv2
from ultralytics import YOLO
import numpy as np
import os
import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class MaskChangeAnalyzer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.results = {}
        
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
    
    def process_video(self, video_path, output_dir=None, frame_interval=10):
        """处理单个视频，每隔frame_interval帧计算一次"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 存储每个track_id的mask序列
        person_masks = defaultdict(list)
        frame_count = 0
        
        print(f"Processing video: {video_name}")
        
        # 逐帧处理视频，每frame_interval帧处理一次
        for results in tqdm(self.model.track(source=video_path, persist=True, stream=True, verbose=False), 
                           desc="Processing frames"):
            frame_count += 1
            
            # 只在指定间隔的帧进行处理
            if frame_count % frame_interval != 0:
                continue
                
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
        
        # 找到主要人物 (基于出现次数和平均面积)
        main_persons = {}
        for track_id, mask_list in person_masks.items():
            if len(mask_list) >= 2:  # 至少出现2次
                avg_area = np.mean([np.sum(mask) for mask in mask_list])
                duration = len(mask_list)
                score = avg_area * duration  # 综合评分
                main_persons[track_id] = {
                    'masks': mask_list,
                    'avg_area': avg_area,
                    'duration': duration,
                    'score': score
                }
        
        # 选择主要人物
        sorted_persons = sorted(main_persons.items(), key=lambda x: x[1]['score'], reverse=True)
        top_persons = dict(sorted_persons[:1])  # 只取最主要的人物
        
        # 计算每个主要人物的变化统计
        analysis_results = {}
        for track_id, person_data in top_persons.items():
            mask_sequence = person_data['masks']
            changes = self.calculate_mask_changes(mask_sequence)
            
            analysis_results[f'person_{track_id}'] = {
                'track_id': track_id,
                'total_keyframes': len(mask_sequence),
                'avg_area': person_data['avg_area'],
                **changes
            }
        
        # 保存结果
        video_result = {
            'video_name': video_name,
            'video_path': video_path,
            'total_frames': frame_count,
            'processed_frames': frame_count // frame_interval,
            'main_persons_count': len(top_persons),
            'persons_analysis': analysis_results
        }
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f'{video_name}_mask_analysis.json'), 'w') as f:
                json.dump(video_result, f, indent=2, default=str)
        
        return video_result
    
    def process_video_for_csv(self, video_path, frame_interval=10):
        """
        为CSV处理优化的视频分析方法，返回汇总统计
        """
        person_masks = defaultdict(list)
        frame_count = 0
        processed_count = 0
        
        # 逐帧处理视频，每frame_interval帧处理一次
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
            return {
                'main_persons_count': 0,
                'avg_person_area': 0,
                'avg_change_ratio': 0.0,
                'max_change_ratio': 0.0,
                'processed_frames': processed_count
            }
        
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
    
    def create_visualization(self, results, output_path):
        """创建可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Person Mask Change Analysis', fontsize=16)
        
        # 收集所有人物的数据
        all_area_changes = []
        all_position_changes = []
        all_shape_changes = []
        person_labels = []
        
        for video_name, video_data in results.items():
            for person_key, person_data in video_data['persons_analysis'].items():
                if 'area_changes_mean' in person_data:
                    all_area_changes.append(person_data['area_changes_mean'])
                    all_position_changes.append(person_data.get('position_changes_mean', 0))
                    all_shape_changes.append(person_data.get('shape_changes_mean', 0))
                    person_labels.append(f"{video_name}_{person_key}")
        
        # 面积变化分布
        if all_area_changes:
            axes[0, 0].hist(all_area_changes, bins=20, alpha=0.7)
            axes[0, 0].set_title('Area Change Distribution')
            axes[0, 0].set_xlabel('Mean Area Change Rate')
            axes[0, 0].set_ylabel('Frequency')
        
        # 位置变化分布
        if all_position_changes:
            axes[0, 1].hist(all_position_changes, bins=20, alpha=0.7, color='orange')
            axes[0, 1].set_title('Position Change Distribution')
            axes[0, 1].set_xlabel('Mean Position Change (pixels)')
            axes[0, 1].set_ylabel('Frequency')
        
        # 形状变化分布
        if all_shape_changes:
            axes[1, 0].hist(all_shape_changes, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Shape Change Distribution')
            axes[1, 0].set_xlabel('Mean Shape Change')
            axes[1, 0].set_ylabel('Frequency')
        
        # 变化类型对比
        if all_area_changes and all_position_changes and all_shape_changes:
            change_data = pd.DataFrame({
                'Area Change': all_area_changes,
                'Position Change': all_position_changes,
                'Shape Change': all_shape_changes
            })
            
            change_data.boxplot(ax=axes[1, 1])
            axes[1, 1].set_title('Change Types Comparison')
            axes[1, 1].set_ylabel('Change Magnitude')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    model_path = '/home/jinqiao/mhi/checkpoints/yolo11n-seg.pt'
    video_dir = '/home/jinqiao/mhi/videos'  # 修改为你的视频目录
    output_dir = '/home/jinqiao/mhi/mask_analysis_results'
    
    analyzer = MaskChangeAnalyzer(model_path)
    
    # 获取视频文件列表
    video_files = []
    if os.path.exists(video_dir):
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # 处理视频
    all_results = {}
    for video_path in video_files[:5]:  # 处理前5个视频作为示例
        try:
            result = analyzer.process_video(video_path, output_dir)
            all_results[result['video_name']] = result
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    # 保存汇总结果
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'summary_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # 创建CSV汇总
    summary_data = []
    for video_name, video_data in all_results.items():
        for person_key, person_data in video_data['persons_analysis'].items():
            row = {
                'video_name': video_name,
                'person_id': person_data['track_id'],
                'total_frames': person_data['total_frames'],
                'avg_area': person_data['avg_area'],
                'duration_ratio': person_data['duration_ratio']
            }
            # 添加变化统计
            for key, value in person_data.items():
                if key not in ['track_id', 'total_frames', 'avg_area', 'duration_ratio']:
                    row[key] = value
            summary_data.append(row)
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(os.path.join(output_dir, 'mask_changes_summary.csv'), index=False)
        
        # 创建可视化
        analyzer.create_visualization(all_results, os.path.join(output_dir, 'mask_changes_visualization.png'))
        
        print(f"Analysis complete! Results saved to {output_dir}")
        print(f"Summary statistics:")
        print(f"- Total videos processed: {len(all_results)}")
        print(f"- Total persons analyzed: {len(summary_data)}")
        print(f"- Average area change: {df_summary['area_changes_mean'].mean():.4f}")
        print(f"- Average position change: {df_summary['position_changes_mean'].mean():.2f} pixels")

if __name__ == '__main__':
    main()