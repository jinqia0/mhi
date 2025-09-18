from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import clip
import os
import cv2
import multiprocessing as mp
from tqdm import tqdm
import time
import queue

# ====== 配置参数 ======
POOL_SIZE_PER_GPU = 2        # 每个GPU的进程数
BATCH_SIZE = 8               # 视频帧批处理大小
SAMPLE_RATE = 10             # 帧采样率

# ====== 路径配置 ======
CSV_DIR = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/csv_1M_aes"  # 修改为你的目录路径
MODEL_PATH = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/weights/aes/sac+logos+ava1-l14-linearMSE.pth"


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def process_video(idx, video_path, aes_model, clip_model, clip_preprocess, device):
    """优化后的视频处理函数（支持批量推理）"""
    if not os.path.exists(video_path):
        return idx, 0.0  # 返回默认值
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_size = max(1, total_frames // SAMPLE_RATE)
    
    scores = []
    frame_buffer = []
    
    with torch.cuda.device(device), torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if (cap.get(cv2.CAP_PROP_POS_FRAMES) % step_size) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buffer.append(Image.fromarray(frame))
                
                # 批量处理
                if len(frame_buffer) >= BATCH_SIZE:
                    # 预处理批次
                    batch = torch.stack([clip_preprocess(img) for img in frame_buffer]).to(device)
                    
                    # 特征提取
                    image_features = clip_model.encode_image(batch)
                    im_emb_arr = normalized(image_features.cpu().numpy())
                    
                    # 推理
                    predictions = aes_model(torch.from_numpy(im_emb_arr).to(device).float())
                    scores.extend(predictions.cpu().tolist())
                    frame_buffer.clear()
        
        # 处理剩余帧
        if len(frame_buffer) > 0:
            batch = torch.stack([clip_preprocess(img) for img in frame_buffer]).to(device)
            image_features = clip_model.encode_image(batch)
            im_emb_arr = normalized(image_features.cpu().numpy())
            predictions = aes_model(torch.from_numpy(im_emb_arr).to(device).float())
            scores.extend(predictions.cpu().tolist())
    
    cap.release()
    return idx, np.mean(scores) if scores else 0.0

class GPUWorker:
    def __init__(self, gpu_id, clip_model_name="ViT-L/14"):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        
        # 初始化模型（保持原始CLIP模型结构）
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.aes_model = MLP(768).to(self.device)
        self.aes_model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.aes_model.eval()
        
        # 添加模型引用（不要使用ModuleDict）
        self.models = {
            'clip': self.clip_model,
            'aes': self.aes_model
        }

    def __call__(self, task_queue, result_queue):
        while True:
            task = task_queue.get()
            if task is None: break
            
            idx, video_path = task
            result = process_video(
                idx, 
                video_path, 
                clip_model=self.models['clip'],  # 显式传递CLIP模型
                aes_model=self.models['aes'],    # 显式传递AES模型
                clip_preprocess=self.clip_preprocess,
                device=self.device
            )
            result_queue.put(result)
            torch.cuda.empty_cache()

def process_single_csv(csv_path):
    """处理单个CSV文件的完整流程，带重试机制"""
    try:
        df = pd.read_csv(csv_path)
        print(f"\n开始处理文件: {csv_path}")
    except Exception as e:
        print(f"文件读取失败: {csv_path} | 错误: {str(e)}")
        return

    video_paths = df['path'].tolist()
    if len(video_paths) == 0:
        print(f"文件无有效视频路径: {csv_path}")
        return

    # ====== 多进程初始化 ======
    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 启动GPU进程
    processes = []
    for gpu_id in range(num_gpus):
        for _ in range(POOL_SIZE_PER_GPU):
            p = mp.Process(target=GPUWorker(gpu_id), args=(task_queue, result_queue))
            p.start()
            processes.append(p)

    # 填充任务队列
    for idx, path in enumerate(video_paths):
        task_queue.put((idx, path))

    # 添加终止信号
    for _ in range(num_gpus * POOL_SIZE_PER_GPU):
        task_queue.put(None)

    # ====== 结果收集 ======
    pbar = tqdm(total=len(video_paths), desc=f"Processing {os.path.basename(csv_path)}")
    results = {}
    retry_count = 0
    max_retries = 5  # 最大重试次数
    while len(results) < len(video_paths):
        try:
            idx, score = result_queue.get(timeout=300)  # 设置超时，防止无限阻塞
            results[idx] = score
            pbar.update(1)
        except queue.Empty:
            retry_count += 1
            print(f"结果队列超时，正在重试... (尝试次数: {retry_count}/{max_retries})")
            if retry_count >= max_retries:
                print("重试次数过多，退出处理")
                break
            time.sleep(5)  # 等待5秒再重试，避免立即重试导致过多负载

    # 更新DataFrame
    df['new_aes'] = df.index.map(lambda x: results.get(x, 0.0))
    
    # 保存结果
    output_path = csv_path.replace('.csv', '_with_aes.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # 清理进程
    for p in processes:
        if p.is_alive():
            p.terminate()
        p.join()

    # 清理队列
    task_queue.close()
    result_queue.close()

def main():
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    print(f"发现 {len(csv_files)} 个待处理CSV文件")
    
    # 依次处理每个文件
    for csv_file in csv_files:
        csv_path = os.path.join(CSV_DIR, csv_file)
        process_single_csv(csv_path)

    print("所有文件处理完成")

if __name__ == "__main__":
    main()
