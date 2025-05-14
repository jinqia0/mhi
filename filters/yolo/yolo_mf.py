import os
import torch
import pandas as pd
import multiprocessing as mp
from ultralytics import YOLO
import cv2

# 配置参数
BATCH_SIZE = 32
CONF_THRESH = 0.6
IOU_THRESH = 0.45
CSV_DIR = "/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/Datasets/csv_1M"
SAVE_INTERVAL = 10000
UPDATE_INTERVAL = 1

# 用于处理每个视频的函数
def process_video(idx, video_path, model, device):
    if not os.path.exists(video_path):
        raise ValueError(f"{video_path} load error, please check the video path.")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_size = max(1, total_frames // 30)
    
    frame_buffer = []
    max_persons = 0
    overlap_flag = 0
    detected = False

    with torch.cuda.device(device):
        with torch.no_grad():
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
                            results = model(frame_buffer, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)

                        for res in results:
                            boxes = res.boxes.xyxy.float()
                            cls_ids = res.boxes.cls
                            
                            # 筛选人体检测
                            person_mask = (cls_ids == 0)
                            person_boxes = boxes[person_mask]
                            person_count = person_mask.sum().item()

                            if person_count > 0:
                                detected = True
                                max_persons = max(max_persons, person_count)

                        frame_buffer.clear()
                        
                        del results
                        torch.cuda.empty_cache()

    cap.release()
    return idx, int(detected), max_persons, overlap_flag

# 处理单个CSV文件的函数
def process_single_csv(csv_path, gpu_id):
    try:
        df = pd.read_csv(csv_path)
        print(f"\n开始处理文件: {csv_path}")
    except Exception as e:
        print(f"文件读取失败: {csv_path} | 错误: {str(e)}")
        return

    # 初始化结果列
    for col in ["has_person", "num_persons", "bbox_overlap"]:
        if col not in df.columns:
            df[col] = 0

    video_paths = df["path"].tolist()
    if len(video_paths) == 0:
        print(f"文件无有效视频路径: {csv_path}")
        return

    # 初始化模型
    model = YOLO("/mnt/pfs-mc0p4k/cvg/team/jinqiao/mhi/weights/yolo/yolo11m.pt ", verbose=False)
    model.to(f"cuda:{gpu_id}")
    model.model.eval()

    # 处理视频
    for idx, video_path in enumerate(video_paths):
        idx, has_person, num_persons, bbox_overlap = process_video(idx, video_path, model, gpu_id)
        df.at[idx, "has_person"] = has_person
        df.at[idx, "num_persons"] = num_persons
        df.at[idx, "bbox_overlap"] = int(bbox_overlap)

    # 保存处理结果
    df.to_csv(csv_path, index=False)
    print(f"处理完成: {csv_path}")

# GPU绑定的进程函数
def gpu_worker(gpu_id, csv_path):
    process_single_csv(csv_path, gpu_id)

if __name__ == "__main__":
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    print(f"发现 {len(csv_files)} 个待处理CSV文件")

    # 启动每个GPU上的进程，处理每个文件
    processes = []
    for i, csv_file in enumerate(csv_files):
        if i >= 8:  # 限制最多8个进程，每个进程绑定到一张GPU
            break
        csv_path = os.path.join(CSV_DIR, csv_file)
        p = mp.Process(target=gpu_worker, args=(i, csv_path))  # 每个进程绑定到一个 GPU
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("所有文件处理完成")
