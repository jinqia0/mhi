import cv2
from ultralytics import YOLO
import itertools
import numpy as np
import sys
import contextlib
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
import pandas as pd

# Import Depth-Anything-V2
sys.path.append('/home/jinqiao/mhi/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

def init_depth_model():
    """初始化深度估计模型"""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    encoder = 'vitb'  # 使用中等大小的模型平衡速度和精度
    depth_model = DepthAnythingV2(**model_configs[encoder])
    
    # 加载预训练权重
    checkpoint_path = f'/home/jinqiao/mhi/checkpoints/depth_anything_v2_{encoder}.pth'
    if os.path.exists(checkpoint_path):
        depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    else:
        print(f"Warning: Depth model checkpoint not found at {checkpoint_path}")
        return None
        
    depth_model = depth_model.to(DEVICE).eval()
    return depth_model

def calculate_mask_depth_stats(mask, depth_map):
    """计算mask区域的深度统计信息"""
    mask_region = mask > 0.5
    if not np.any(mask_region):
        return None
    
    depth_values = depth_map[mask_region]
    return {
        'mean_depth': np.mean(depth_values),
        'median_depth': np.median(depth_values),
        'std_depth': np.std(depth_values),
        'min_depth': np.min(depth_values),
        'max_depth': np.max(depth_values)
    }

def is_depth_similar(depth_stats1, depth_stats2, similarity_threshold=0.2):
    """判断两个人物的深度是否相近
    
    Args:
        depth_stats1, depth_stats2: 深度统计信息字典
        similarity_threshold: 相似度阈值，越小越严格
    
    Returns:
        bool: 是否深度相近
    """
    if depth_stats1 is None or depth_stats2 is None:
        return False
    
    # 使用中位数深度进行比较，更鲁棒
    depth1 = depth_stats1['median_depth']
    depth2 = depth_stats2['median_depth']
    
    # 计算相对深度差异
    avg_depth = (depth1 + depth2) / 2
    if avg_depth == 0:
        return abs(depth1 - depth2) < similarity_threshold
    
    relative_diff = abs(depth1 - depth2) / avg_depth
    
    # 额外考虑深度标准差，如果两个人物内部深度变化很大，放宽要求
    std_factor = max(depth_stats1['std_depth'], depth_stats2['std_depth']) / avg_depth
    adjusted_threshold = similarity_threshold + std_factor * 0.1
    
    return relative_diff < adjusted_threshold

def has_depth_contact(mask1, mask2, depth_map, contact_depth_threshold=0.15):
    """判断两个重叠区域是否在深度上构成接触
    
    Args:
        mask1, mask2: 人物mask
        depth_map: 深度图
        contact_depth_threshold: 接触深度阈值
    
    Returns:
        bool: 是否构成深度接触
    """
    # 找到重叠区域
    intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5)
    if not np.any(intersection):
        return False
    
    # 获取重叠区域周围的深度信息
    # 扩展重叠区域来获取更多上下文
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_intersection = cv2.dilate(intersection.astype(np.uint8), kernel, iterations=2)
    
    # 分别获取两个人物在扩展重叠区域的深度
    mask1_contact = np.logical_and(mask1 > 0.5, expanded_intersection > 0)
    mask2_contact = np.logical_and(mask2 > 0.5, expanded_intersection > 0)
    
    if not (np.any(mask1_contact) and np.any(mask2_contact)):
        return False
    
    depth1_contact = depth_map[mask1_contact]
    depth2_contact = depth_map[mask2_contact]
    
    # 计算接触区域的深度差异
    median_depth1 = np.median(depth1_contact)
    median_depth2 = np.median(depth2_contact)
    
    avg_depth = (median_depth1 + median_depth2) / 2
    if avg_depth == 0:
        return abs(median_depth1 - median_depth2) < contact_depth_threshold
    
    relative_depth_diff = abs(median_depth1 - median_depth2) / avg_depth
    
    return relative_depth_diff < contact_depth_threshold

def process_video_with_depth(video_path, model_path, video_dir, gpu_id, mhi_root, depth_model):
    """处理单个视频，集成深度信息的接触判定"""
    import os
    import sys
    import contextlib
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    from ultralytics import YOLO
    
    frame_count = 0
    contact_frame_set = set()  # 原始mask重叠
    depth_contact_frame_set = set()  # 深度接触
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join('runs', 'track', video_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 打开视频文件用于深度估计
    cap = cv2.VideoCapture(video_path)
    
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        yolo_results = YOLO(model_path).track(
            source=video_path, 
            persist=True, 
            stream=True, 
            verbose=False, 
            save=True, 
            project='runs/track', 
            name=video_name, 
            exist_ok=True
        )
        
        for results in yolo_results:
            frame_count += 1
            
            # 读取当前帧用于深度估计
            ret, frame = cap.read()
            if not ret:
                break
                
            # 获取深度图
            depth_map = None
            if depth_model is not None:
                try:
                    with torch.no_grad():
                        depth_map = depth_model.infer_image(frame, input_size=518)
                except Exception as e:
                    print(f"深度估计失败 {video_name} frame {frame_count}: {e}")
            
            masks_list = []
            track_ids_list = []
            areas_list = []
            
            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.arange(len(masks))
                    
                    for mask, cls, track_id in zip(masks, classes, ids):
                        if cls == 0:  # 人物类别
                            mask_bin = (mask > 0.5).astype(np.uint8)
                            area = np.sum(mask_bin)
                            masks_list.append(mask_bin)
                            track_ids_list.append(track_id)
                            areas_list.append(area)
            
            # 主要人物判定
            if areas_list:
                max_area = max(areas_list)
                main_person_indices = [i for i, area in enumerate(areas_list) if area >= max_area * 0.5]
                
                # 只计算主要人物之间的交集
                contact_this_frame = False
                depth_contact_this_frame = False
                
                for (i, mask1_idx), (j, mask2_idx) in itertools.combinations(enumerate(main_person_indices), 2):
                    mask1 = masks_list[mask1_idx]
                    mask2 = masks_list[mask2_idx]
                    
                    # 原始mask重叠判定
                    intersection = np.logical_and(mask1, mask2)
                    if np.sum(intersection) > 0:
                        contact_this_frame = True
                        
                        # 深度接触判定
                        if depth_map is not None:
                            # 方法1：判断两个人物整体深度是否相近
                            depth_stats1 = calculate_mask_depth_stats(mask1, depth_map)
                            depth_stats2 = calculate_mask_depth_stats(mask2, depth_map)
                            depth_similar = is_depth_similar(depth_stats1, depth_stats2)
                            
                            # 方法2：判断接触区域深度是否连续
                            contact_depth_continuous = has_depth_contact(mask1, mask2, depth_map)
                            
                            # 两种方法任一满足即认为是深度接触
                            if depth_similar or contact_depth_continuous:
                                depth_contact_this_frame = True
                
                if contact_this_frame:
                    contact_frame_set.add(frame_count)
                if depth_contact_this_frame:
                    depth_contact_frame_set.add(frame_count)
    
    cap.release()
    
    # 计算统计结果
    rel_path = os.path.relpath(video_path, mhi_root)
    
    # 原始接触统计
    contact_frames = len(contact_frame_set)
    contact_frame_ratio = contact_frames / frame_count if frame_count > 0 else 0
    is_contact = 1 if contact_frame_ratio >= 0.1 else 0
    
    # 深度接触统计
    depth_contact_frames = len(depth_contact_frame_set)
    depth_contact_frame_ratio = depth_contact_frames / frame_count if frame_count > 0 else 0
    is_depth_contact = 1 if depth_contact_frame_ratio >= 0.1 else 0
    
    return [[
        rel_path, 
        contact_frames, 
        depth_contact_frames,
        frame_count, 
        f"{contact_frame_ratio:.3f}", 
        f"{depth_contact_frame_ratio:.3f}",
        is_contact,
        is_depth_contact
    ]]

if __name__ == '__main__':
    model_path = '/home/jinqiao/mhi/checkpoints/yolo11n-seg.pt'
    mhi_root = '/home/jinqiao/mhi'
    video_dir = '/home/jinqiao/mhi/Dataset/panda'
    
    # 初始化深度模型
    print("初始化深度估计模型...")
    depth_model = init_depth_model()
    if depth_model is None:
        print("深度模型加载失败，请检查模型文件路径")
        sys.exit(1)
    
    # 获取视频文件列表
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    video_files = sorted(video_files)[:100]  # 限制处理数量进行测试
    
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    
    csv_path = '../results/contact_stats_with_depth.csv'
    all_is_contact = []
    all_is_depth_contact = []
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'video', 
            'contact_frames', 
            'depth_contact_frames', 
            'total_frames', 
            'contact_frame_ratio', 
            'depth_contact_frame_ratio',
            'is_contact',
            'is_depth_contact'
        ])
        
        # 注意：由于深度模型需要GPU，这里减少并发数避免显存不足
        max_workers = min(num_gpus * 2, 4)  # 限制并发数
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, video_path in enumerate(video_files):
                gpu_id = idx % num_gpus
                futures.append(executor.submit(
                    process_video_with_depth, 
                    video_path, 
                    model_path, 
                    video_dir, 
                    gpu_id, 
                    mhi_root,
                    depth_model
                ))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='处理视频'):
                try:
                    for row in future.result():
                        writer.writerow(row)
                        all_is_contact.append(row[-2])
                        all_is_depth_contact.append(row[-1])
                except Exception as e:
                    print(f"处理视频时出错: {e}")
    
    # 统计结果
    total_videos = len(video_files)
    contact_videos = sum(1 for x in all_is_contact if x == 1)
    depth_contact_videos = sum(1 for x in all_is_depth_contact if x == 1)
    
    contact_ratio = contact_videos / total_videos if total_videos > 0 else 0
    depth_contact_ratio = depth_contact_videos / total_videos if total_videos > 0 else 0
    
    print(f'统计结果已保存到 {csv_path}')
    print(f'原始mask重叠视频: {contact_videos} / {total_videos}，比率: {contact_ratio:.3f}')
    print(f'深度接触视频: {depth_contact_videos} / {total_videos}，比率: {depth_contact_ratio:.3f}')
    print(f'深度过滤效果: 过滤掉 {contact_videos - depth_contact_videos} 个误检视频')