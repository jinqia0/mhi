#!/usr/bin/env python3
"""
测试脚本：验证深度接触判定算法
处理单个视频文件，输出详细的调试信息
"""

import cv2
from ultralytics import YOLO
import itertools
import numpy as np
import sys
import contextlib
import os
import torch
import matplotlib.pyplot as plt

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
    
    encoder = 'vits'
    depth_model = DepthAnythingV2(**model_configs[encoder])
    
    checkpoint_path = f'depth_anything_v2_{encoder}.pth'
    if os.path.exists(checkpoint_path):
        depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"成功加载深度模型: {encoder}")
    else:
        print(f"警告: 深度模型权重文件未找到 {checkpoint_path}")
        return None
        
    depth_model = depth_model.to(DEVICE).eval()
    return depth_model

def visualize_depth_contact(frame, masks_list, depth_map, frame_num, output_dir, contact_pairs=None):
    """可视化深度接触分析结果，支持多人物场景"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_people = len(masks_list)
    fig_height = 12 if num_people <= 3 else 16
    fig, axes = plt.subplots(2, 3, figsize=(18, fig_height))
    
    # 原始帧
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Original Frame (Frame {frame_num})')
    axes[0, 0].axis('off')
    
    # 深度图
    depth_colored = plt.cm.viridis(depth_map / depth_map.max())
    axes[0, 1].imshow(depth_colored)
    axes[0, 1].set_title('Depth Map')
    axes[0, 1].axis('off')
    
    # 人物masks - 优化多人显示
    combined_mask = np.zeros_like(depth_map, dtype=np.float32)
    colors = np.linspace(1, num_people, num_people)
    
    for i, mask in enumerate(masks_list):
        combined_mask[mask > 0.5] = colors[i]
    
    axes[0, 2].imshow(combined_mask, cmap='tab10', vmin=0, vmax=num_people)
    axes[0, 2].set_title(f'Person Masks ({num_people} people)')
    axes[0, 2].axis('off')
    
    # 接触区域分析 - 改进多人物对可视化
    all_contacts = []
    contact_viz = np.zeros((frame.shape[0], frame.shape[1], 3))
    
    # 为每个人分配独特颜色（基础显示）
    person_colors = plt.cm.Set3(np.linspace(0, 1, num_people))[:, :3]
    
    # 显示所有人物（基础层）
    for i, mask in enumerate(masks_list):
        contact_viz[mask > 0.5] = person_colors[i] * 0.4  # 更淡的基础色
    
    # 处理接触对 - 改进多对可视化
    contact_pair_info = []
    if contact_pairs:
        # 使用更明显的颜色区分不同接触对
        contact_colors = plt.cm.rainbow(np.linspace(0, 1, len(contact_pairs)))
        
        for idx, (pair_info, color) in enumerate(zip(contact_pairs, contact_colors)):
            mask_i = pair_info.get('mask1_idx', 0) 
            mask_j = pair_info.get('mask2_idx', 1)
            
            if mask_i < len(masks_list) and mask_j < len(masks_list):
                mask1, mask2 = masks_list[mask_i], masks_list[mask_j]
                intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5)
                
                if np.any(intersection):
                    # 使用更鲜明的颜色高亮接触区域
                    contact_viz[intersection] = color[:3] * 0.9
                    contact_depths = depth_map[intersection]
                    all_contacts.extend(contact_depths)
                    
                    # 记录接触对信息用于标题显示
                    pair_num = pair_info.get('pair', (mask_i+1, mask_j+1))
                    depth_similar = pair_info.get('depth_similar', False)
                    overlap_pixels = pair_info.get('overlap_pixels', np.sum(intersection))
                    contact_pair_info.append({
                        'pair': pair_num,
                        'pixels': overlap_pixels,
                        'depth_contact': depth_similar
                    })
    
    # 显示接触可视化 - 改进标题信息
    axes[1, 1].imshow(contact_viz)
    title = f'Contact Visualization'
    if contact_pairs:
        depth_contacts = sum(1 for info in contact_pair_info if info['depth_contact'])
        title += f'\n{len(contact_pairs)} pairs ({depth_contacts} depth contacts)'
        
        # 在图上添加文字说明
        y_pos = 0.95
        for info in contact_pair_info:
            pair_text = f"Pair {info['pair'][0]}-{info['pair'][1]}: {info['pixels']}px"
            if info['depth_contact']:
                pair_text += " ✓"
            axes[1, 1].text(0.02, y_pos, pair_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            y_pos -= 0.08
    
    axes[1, 1].set_title(title)
    axes[1, 1].axis('off')
    
    # 接触区域深度分布
    if all_contacts:
        axes[1, 0].hist(all_contacts, bins=30, alpha=0.7, color='orange', label='All Contact Regions')
        axes[1, 0].set_title('Contact Regions Depth Distribution')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Depth Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # 添加统计信息
        contact_mean = np.mean(all_contacts)
        contact_std = np.std(all_contacts)
        axes[1, 0].axvline(contact_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {contact_mean:.2f}')
        axes[1, 0].axvline(contact_mean + contact_std, color='red', linestyle=':', alpha=0.5)
        axes[1, 0].axvline(contact_mean - contact_std, color='red', linestyle=':', alpha=0.5)
    else:
        axes[1, 0].text(0.5, 0.5, f'No Contact Detected\n({num_people} people total)', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].axis('off')
    
    # 深度接触热力图
    if all_contacts:
        contact_heatmap = np.zeros_like(depth_map)
        if contact_pairs:
            for pair_info in contact_pairs:
                mask_i = pair_info.get('mask1_idx', 0)
                mask_j = pair_info.get('mask2_idx', 1)
                
                if mask_i < len(masks_list) and mask_j < len(masks_list):
                    mask1, mask2 = masks_list[mask_i], masks_list[mask_j]
                    intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5)
                    
                    if np.any(intersection):
                        # 计算接触区域的深度一致性
                        contact_depths = depth_map[intersection]
                        depth_consistency = 1.0 / (1.0 + np.std(contact_depths))
                        contact_heatmap[intersection] = depth_consistency
        
        axes[1, 2].imshow(contact_heatmap, cmap='hot', alpha=0.8)
        axes[1, 2].set_title('Contact Depth Consistency\n(Brighter = More Similar)')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'No Contact for\nConsistency Analysis', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 2].transAxes, fontsize=14)
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'frame_{frame_num:04d}_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

def is_depth_similar_at_contact(mask1, mask2, depth_map, similarity_threshold=0.01):
    """判断两个人物在接触点处的深度是否相近"""
    # 找到接触区域（两个mask的交集）
    intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5)
    
    if not np.any(intersection):
        return False, "没有接触区域"
    
    contact_pixels = np.sum(intersection)
    
    # 获取接触区域附近两个人物的深度
    # 扩展接触区域以获得更稳定的深度估计
    kernel = np.ones((5, 5), np.uint8)
    expanded_contact = cv2.dilate(intersection.astype(np.uint8), kernel, iterations=1)
    
    # 获取两个人物在扩展接触区域的深度
    person1_region = np.logical_and(mask1 > 0.5, expanded_contact.astype(bool))
    person2_region = np.logical_and(mask2 > 0.5, expanded_contact.astype(bool))
    
    if not np.any(person1_region) or not np.any(person2_region):
        raise ValueError("无法获取接触区域附近的人物深度数据")
    
    # 获取两个人物的深度值
    depth1_values = depth_map[person1_region]
    depth2_values = depth_map[person2_region]
    
    # 使用中位数减少噪声影响
    depth1_median = np.median(depth1_values)
    depth2_median = np.median(depth2_values)
    
    # 检查深度数据有效性
    if depth1_median <= 0 or depth2_median <= 0:
        raise ValueError(f"深度数据异常: 人物1深度={depth1_median:.3f}, 人物2深度={depth2_median:.3f}")
    
    # 计算相对深度差异（相对于全局深度范围）
    global_depth_range = np.max(depth_map) - np.min(depth_map)
    depth_diff = abs(depth1_median - depth2_median)
    
    if global_depth_range <= 0:
        raise ValueError(f"全局深度范围异常: {global_depth_range:.3f}")
    
    relative_diff = depth_diff / global_depth_range
    is_similar = relative_diff < similarity_threshold
    
    reason = f"人物深度差异: {relative_diff:.3f} (={depth_diff:.3f}/{global_depth_range:.3f}), 阈值: {similarity_threshold:.3f}, 接触像素: {contact_pixels}, 深度: {depth1_median:.3f} vs {depth2_median:.3f}"
    return is_similar, reason

def test_single_video(video_path, output_dir='debug_output', sample_frames=10):
    """测试单个视频的深度接触判定"""
    print(f"测试视频: {video_path}")
    
    # 初始化模型
    depth_model = init_depth_model()
    if depth_model is None:
        print("深度模型初始化失败")
        return
    
    model_path = '/home/jinqiao/mhi/checkpoints/yolo11n-seg.pt'
    yolo_model = YOLO(model_path)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    
    # 采样帧进行分析
    sample_indices = np.linspace(0, total_frames-1, min(sample_frames, total_frames), dtype=int)
    
    contact_results = []
    
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        print(f"\n处理帧 {frame_idx+1}/{total_frames}")
        
        # YOLO检测
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results = yolo_model(frame, verbose=False)
        
        masks_list = []
        areas_list = []
        
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for mask, cls in zip(masks, classes):
                    if cls == 0:  # 人物
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_bin = (mask_resized > 0.5).astype(np.uint8)
                        area = np.sum(mask_bin)
                        if area > 1000:  # 过滤太小的检测
                            masks_list.append(mask_bin)
                            areas_list.append(area)
        
        print(f"检测到 {len(masks_list)} 个人物")
        
        if len(masks_list) < 2:
            print("人物数量不足，跳过")
            continue
        
        # 深度估计
        with torch.no_grad():
            depth_map = depth_model.infer_image(frame, input_size=518)
        
        # 找主要人物
        max_area = max(areas_list)
        main_person_indices = [i for i, area in enumerate(areas_list) if area >= max_area * 0.5]
        print(f"主要人物数量: {len(main_person_indices)}")
        
        # 分析所有人物对的接触情况
        frame_contacts = []
        contact_count = 0
        depth_contact_count = 0
        
        for (i, mask1_idx), (j, mask2_idx) in itertools.combinations(enumerate(main_person_indices), 2):
            mask1 = masks_list[mask1_idx]
            mask2 = masks_list[mask2_idx]
            
            # 检查接触
            intersection = np.logical_and(mask1, mask2)
            overlap_pixels = np.sum(intersection)
            
            if overlap_pixels > 0:
                contact_count += 1
                # 深度相似性判定
                is_similar, reason = is_depth_similar_at_contact(mask1, mask2, depth_map)
                
                if is_similar:
                    depth_contact_count += 1
                    
                print(f"  人物对 {i+1}-{j+1}: 接触像素 {overlap_pixels}, 深度相似: {is_similar}")
                print(f"    详情: {reason}")
                
                frame_contacts.append({
                    'frame_idx': frame_idx,
                    'pair': (i+1, j+1),
                    'mask1_idx': mask1_idx,
                    'mask2_idx': mask2_idx,
                    'overlap_pixels': overlap_pixels,
                    'depth_similar': is_similar,
                    'reason': reason
                })
            else:
                print(f"  人物对 {i+1}-{j+1}: 无接触像素")

        # 帧级别总结
        if contact_count > 0:
            print(f"  帧总结: {contact_count} 个接触对, {depth_contact_count} 个深度接触对")
        
        # 可视化结果 - 传递接触对信息
        visualize_depth_contact(frame, masks_list, depth_map, frame_idx, output_dir, frame_contacts)
        
        contact_results.extend(frame_contacts)
    
    cap.release()
    
    # 详细统计总结
    print(f"\n=== 分析总结 ===")
    print(f"总采样帧数: {len(sample_indices)}")
    
    if contact_results:
        contact_frames = set(c['frame_idx'] for c in contact_results)
        depth_contact_frames = set(c['frame_idx'] for c in contact_results if c['depth_similar'])
        
        total_contact_pairs = len(contact_results)
        depth_contact_pairs = sum(1 for c in contact_results if c['depth_similar'])
        
        print(f"有接触的帧数: {len(contact_frames)}")
        print(f"有深度接触的帧数: {len(depth_contact_frames)}")
        print(f"总接触对数: {total_contact_pairs}")
        print(f"深度接触对数: {depth_contact_pairs}")
        print(f"深度接触比例: {depth_contact_pairs/total_contact_pairs:.3f}")
        
        # 按帧统计
        frame_stats = {}
        for result in contact_results:
            frame_idx = result['frame_idx']
            if frame_idx not in frame_stats:
                frame_stats[frame_idx] = {'total': 0, 'depth_contact': 0}
            frame_stats[frame_idx]['total'] += 1
            if result['depth_similar']:
                frame_stats[frame_idx]['depth_contact'] += 1
        
        print(f"\n按帧接触统计:")
        for frame_idx in sorted(frame_stats.keys()):
            stats = frame_stats[frame_idx]
            ratio = stats['depth_contact'] / stats['total']
            print(f"  帧 {frame_idx}: {stats['depth_contact']}/{stats['total']} 深度接触 ({ratio:.2f})")
    else:
        print("未检测到任何人物接触")
    
    return contact_results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python test_depth_contact.py <video_path> [output_dir]")
        print("例如: python test_depth_contact.py /path/to/video.mp4 debug_output")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'debug_output'
    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0])
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        sys.exit(1)
    
    results = test_single_video(video_path, output_dir, sample_frames=5)
    print(f"\n调试输出保存在: {output_dir}")