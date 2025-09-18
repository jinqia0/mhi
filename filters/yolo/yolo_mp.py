import cv2
import os
import torch
import pandas as pd
import multiprocessing as mp
from ultralytics import YOLO
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
torch.backends.cudnn.benchmark = True  # 启用cuDNN优化

# ====== 配置参数 ======
BATCH_SIZE = 32              # 根据显存调整（建议RTX3090设为4，A100设为8）
CONF_THRESH = 0.6            # 提高置信度阈值减少检测量
IOU_THRESH = 0.45            # 调整IoU阈值
POOL_SIZE_PER_GPU = 4        # 每个GPU的进程数（根据CPU核心数调整）
UPDATE_INTERVAL= 1          # 进度条更新间隔
SAVE_INTERVAL = 10000          # 新增：每处理100个视频保存一次结果

# ====== 路径配置 ======
csv_path = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/csv_1M/panda_all_text_aes_part_20.csv"

# ====== 初始化数据 ======
df = pd.read_csv(csv_path)
video_paths = df["path"].tolist()

# 初始化结果列
for col in ["has_person", "num_persons", "bbox_overlap"]:
    if col not in df.columns:
        df[col] = 0

# ====== 多GPU处理核心 ======
def calculate_iou(box1, box2):
    """支持三维张量的IoU计算（正确处理广播维度）"""
    # 确保内存连续性并保持维度
    box1 = box1.contiguous().view(-1,4)  # 转换为(N*M,4)
    box2 = box2.contiguous().view(-1,4)
    
    # 维度验证
    assert box1.shape == box2.shape, f"Box shapes mismatch: {box1.shape} vs {box2.shape}"
    
    # 计算交集坐标
    inter_x1 = torch.max(box1[:,0], box2[:,0])
    inter_y1 = torch.max(box1[:,1], box2[:,1])
    inter_x2 = torch.min(box1[:,2], box2[:,2])
    inter_y2 = torch.min(box1[:,3], box2[:,3])
    
    # 计算面积
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])
    
    # 恢复原始矩阵形状
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou.view(box1.size(0)//box2.size(0), box2.size(0))  # 恢复为(N,M)形状

def process_video(idx, video_path, model, device):
    """ 优化后的视频处理（支持批量推理） """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_size = max(1, total_frames // 30)
        
        frame_buffer = []
        max_persons = 0
        overlap_flag = 0
        detected = False

        with torch.cuda.device(device):  # 显式指定设备上下文
            with torch.no_grad():        # 禁用梯度计算
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    # 动态帧采样策略
                    if (cap.get(cv2.CAP_PROP_POS_FRAMES) % step_size) == 0:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_buffer.append(frame)

                        # 批量推理（达到批次大小或视频结束）
                        if len(frame_buffer) == BATCH_SIZE or not ret:
                            with torch.inference_mode():
                                results = model(frame_buffer, 
                                            conf=CONF_THRESH, 
                                            iou=IOU_THRESH, 
                                            verbose=False)

                            for res in results:
                                # 同步张量到CPU并转换类型
                                boxes = res.boxes.xyxy.float()
                                cls_ids = res.boxes.cls
                                
                                # 筛选人体检测
                                person_mask = (cls_ids == 0)
                                person_boxes = boxes[person_mask]
                                person_count = person_mask.sum().item()

                                # 关键修复：更新最大人数统计
                                if person_count > 0:
                                    detected = True
                                    max_persons = max(max_persons, person_count)

                                # 批量计算IoU（使用矩阵运算）
                                if person_boxes.shape[0] >= 2:
                                    n = person_boxes.shape[0]
                                    iou_matrix = calculate_iou(
                                        person_boxes.unsqueeze(1),
                                        person_boxes.unsqueeze(0)
                                    )
                                    overlap_flag = max(overlap_flag, (iou_matrix > 0).any().item())

                            frame_buffer.clear()

        cap.release()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return idx, -1, -1, -1
    return idx, int(detected), max_persons, overlap_flag

def gpu_worker(gpu_id, task_queue, result_queue):
    """ GPU绑定的长期工作进程 """
    torch.cuda.set_device(gpu_id)
    model = YOLO("/home/jinqiao/mhi/checkpoints/yolo11m.pt", verbose=False)  # 修复模型版本
    model.to(f"cuda:{gpu_id}")
    model.model.eval()  # 设置为评估模式
    
    # 优化显存配置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    while True:
        task = task_queue.get()
        if task is None: break  # 终止信号
        result = process_video(*task, model=model, device=gpu_id)
        result_queue.put(result)
        del result
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    # ====== 初始化并行环境 ======
    mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    
    # ====== 创建任务队列 ======
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # ====== 启动GPU工作进程 ======
    for gpu_id in range(num_gpus):
        for _ in range(POOL_SIZE_PER_GPU):  # 每个GPU启动多个进程
            mp.Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue)).start()

    # ====== 填充任务队列 ======
    for idx, path in enumerate(video_paths):
        task_queue.put((idx, path))

    # ====== 添加终止信号 ======
    for _ in range(num_gpus * POOL_SIZE_PER_GPU):
        task_queue.put(None)

    # ====== 结果收集 ====== 
    pbar = tqdm(total=len(video_paths), desc="Processing")
    completed = 0
    count = 0
    while completed < len(video_paths):
        idx, has_person, num_persons, bbox_overlap = result_queue.get()
        df.at[idx, "has_person"] = has_person
        df.at[idx, "num_persons"] = num_persons
        df.at[idx, "bbox_overlap"] = int(bbox_overlap)
        completed += 1
        
        # # 新增：定期保存逻辑
        # if completed % SAVE_INTERVAL == 0:
        #     df.to_csv(csv_path, index=False)
        #     pbar.write(f"自动保存：已处理 {completed} 个视频")
            
        # 原进度条更新逻辑
        if completed % UPDATE_INTERVAL == 0 or completed == len(video_paths):
            update_count = UPDATE_INTERVAL if completed % UPDATE_INTERVAL == 0 else completed % UPDATE_INTERVAL
            pbar.update(update_count)
            pbar.set_postfix_str(f"GPU Utilization: {torch.cuda.utilization()}%")
            
        if (num_persons >= 2):
            count += 1
    
    # 最终保存确保数据完整
    df.to_csv(csv_path, index=False)
    print(count / len(video_paths))
    print(f"处理完成，最终文件已保存: {csv_path}")