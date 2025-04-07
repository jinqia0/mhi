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

# ====== 配置参数 ======
POOL_SIZE_PER_GPU = 2        # 每个GPU的进程数
BATCH_SIZE = 8               # 视频帧批处理大小
SAMPLE_RATE = 10             # 帧采样率

# ====== 路径配置 ======
csv_path = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/panda_10k.csv"
model_path = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/weights/aes/sac+logos+ava1-l14-linearMSE.pth"

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

def process_video(idx, video_path, model, clip_preprocess, device):
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
                    image_features = model.encode_image(batch)
                    im_emb_arr = normalized(image_features.cpu().numpy())
                    
                    # 推理
                    predictions = model.aes_model(torch.from_numpy(im_emb_arr).to(device).float())
                    scores.extend(predictions.cpu().tolist())
                    frame_buffer.clear()
        
        # 处理剩余帧
        if len(frame_buffer) > 0:
            batch = torch.stack([clip_preprocess(img) for img in frame_buffer]).to(device)
            image_features = model.encode_image(batch)
            im_emb_arr = normalized(image_features.cpu().numpy())
            predictions = model.aes_model(torch.from_numpy(im_emb_arr).to(device).float())
            scores.extend(predictions.cpu().tolist())
    
    cap.release()
    return idx, np.mean(scores) if scores else 0.0

class GPUWorker:
    def __init__(self, gpu_id, clip_model_name="ViT-L/14"):
        self.gpu_id = gpu_id
        self.clip_model_name = clip_model_name
        self.device = torch.device(f"cuda:{gpu_id}")
        
        # 初始化模型
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.aes_model = MLP(768).to(self.device)
        self.aes_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.aes_model.eval()
        
        # 合并模型以简化参数传递
        self.model = nn.ModuleDict({
            'clip': self.clip_model,
            'aes_model': self.aes_model
        }).eval()

    def __call__(self, task_queue, result_queue):
        while True:
            task = task_queue.get()
            if task is None: break  # 终止信号
            
            idx, video_path = task
            result = process_video(idx, video_path, self.model, self.clip_preprocess, self.device)
            result_queue.put(result)
            torch.cuda.empty_cache()

def main():
    # 初始化并行环境
    mp.set_start_method('spawn')
    df = pd.read_csv(csv_path)
    video_paths = df['path'].tolist()
    num_gpus = torch.cuda.device_count()
    
    # 创建任务队列
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # 启动GPU工作进程
    workers = []
    for gpu_id in range(num_gpus):
        for _ in range(POOL_SIZE_PER_GPU):
            worker = mp.Process(
                target=GPUWorker(gpu_id),
                args=(task_queue, result_queue)
            )
            worker.start()
            workers.append(worker)
    
    # 填充任务队列
    for idx, path in enumerate(video_paths):
        task_queue.put((idx, path))
    
    # 添加终止信号
    for _ in range(num_gpus * POOL_SIZE_PER_GPU):
        task_queue.put(None)
    
    # 结果收集
    pbar = tqdm(total=len(video_paths), desc="Processing Videos")
    results = {}
    while len(results) < len(video_paths):
        idx, score = result_queue.get()
        results[idx] = score
        pbar.update(1)
        pbar.set_postfix_str(f"Current Score: {score:.2f}")
    
    # 更新DataFrame
    df['new_aes'] = df.index.map(lambda x: results.get(x, 0.0))
    
    # 保存结果
    output_path = csv_path.replace('.csv', '_with_aes.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
