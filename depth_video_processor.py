#!/usr/bin/env python3
import cv2
import torch
import numpy as np
import sys
import os
from pathlib import Path

# 添加第三方库路径
sys.path.append(str(Path(__file__).parent / 'third_party' / 'Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2

class DepthVideoProcessor:
    def __init__(self, model_configs='vits', device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 模型配置
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        # 初始化深度估计模型
        encoder = 'vits'
        self.model = DepthAnythingV2(**model_configs[encoder])
        
        # 加载模型权重
        checkpoint_path = 'checkpoints/depth_anything_v2_vits.pth'
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
        self.model = self.model.to(self.device).eval()
        
    def process_frame(self, frame):
        """处理单帧图像，返回深度图"""
        with torch.no_grad():
            # 预处理
            original_h, original_w = frame.shape[:2]
            
            # 调整尺寸为14的倍数
            target_h = (original_h // 14) * 14
            target_w = (original_w // 14) * 14
            
            # 如果需要调整尺寸
            if target_h != original_h or target_w != original_w:
                frame_resized = cv2.resize(frame, (target_w, target_h))
            else:
                frame_resized = frame
            
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb / 255.0
            
            # 转换为tensor
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            
            # 深度估计
            depth = self.model(frame_tensor)
            
            # 处理输出
            depth = depth.squeeze().cpu().numpy()
            
            # 归一化深度图到0-255范围
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            
            # 如果之前调整过尺寸，需要调整回原始尺寸
            if target_h != original_h or target_w != original_w:
                depth_colored = cv2.resize(depth_colored, (original_w, original_h))
            
            return depth_colored
    
    def process_video_with_concatenation(self, input_video_path, output_video_path):
        """处理视频并将原视频与深度图水平拼接"""
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"处理视频: {input_video_path}")
        print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
        
        # 设置输出视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理深度估计
                depth_frame = self.process_frame(frame)
                
                # 确保深度图与原图尺寸一致
                if depth_frame.shape[:2] != frame.shape[:2]:
                    depth_frame = cv2.resize(depth_frame, (width, height))
                
                # 水平拼接原图和深度图
                concatenated_frame = np.hstack([frame, depth_frame])
                
                # 写入输出视频
                out.write(concatenated_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"已处理帧数: {frame_count}/{total_frames}")
        
        finally:
            cap.release()
            out.release()
        
        print(f"视频处理完成，输出保存至: {output_video_path}")
        return output_video_path

def main():
    # 设置输入输出路径
    input_video = "enhancement_samples/depth/new_correct_negatives/-0vNoHB2ueI-0:07:18.604-0:07:21.482.mp4"
    output_video = "results/-0vNoHB2ueI-depth_comparison.mp4"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # 初始化处理器
    processor = DepthVideoProcessor()
    
    # 处理视频
    try:
        result_path = processor.process_video_with_concatenation(input_video, output_video)
        print(f"成功生成拼接视频: {result_path}")
    except Exception as e:
        print(f"处理视频时发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())