import cv2
import os
import torch
import pandas as pd
import multiprocessing as mp
from ultralytics import YOLO
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ====== 配置路径 ======
csv_path = "./panda_10k_interaction_score.csv"  # 直接修改原 CSV
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

# 读取 CSV
df = pd.read_csv(csv_path)

# 初始化新列
if "has_person" not in df.columns:
    df["has_person"] = 0
if "num_persons" not in df.columns:
    df["num_persons"] = 0
if "bbox_overlap" not in df.columns:
    df["bbox_overlap"] = 0  # 新增列，标记边界框是否重叠

# 获取所有可用 GPU
num_gpus = torch.cuda.device_count()
gpu_ids = list(range(num_gpus))  # 例如 [0, 1, 2, 3]

# 分配任务
video_paths = df["path"].tolist()
num_videos = len(video_paths)

# 设定全局变量，方便多进程修改 CSV
csv_lock = mp.Lock()

def calculate_iou(box1, box2):
    """ 计算两个边界框的 IoU """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def process_video(idx, video_path, gpu_id):
    """ 在指定 GPU 上运行 YOLO 进行人体检测，并计算边界框重叠 """
    if not os.path.exists(video_path):
        return idx, 0, 0, 0  # 文件不存在，返回 0

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置检测间隔，确保长视频检测帧数多
    step_size = max(1, total_frames // 30)
    frame_idx = 0
    detected_persons = set()
    bbox_overlap = 0  # 记录是否有重叠的边界框

    # 加载 YOLOv8 模型到指定 GPU
    model = YOLO("yolo11n.pt", verbose=False)
    model.to(f"cuda:{gpu_id}")  # 绑定到指定 GPU

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 仅检测首帧 & 每 step_size 帧
        if frame_idx == 0 or frame_idx % step_size == 0:
            results = model.predict(frame, verbose=False, device=f"cuda:{gpu_id}")
            
            person_bboxes = []
            person_count = 0

            for box in results[0].boxes:
                if int(box.cls[0].item()) == 0:  # YOLO 类别 0 代表人体
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取边界框
                    person_bboxes.append((x1, y1, x2, y2))
                    person_count += 1

            if person_count > 0:
                detected_persons.add(person_count)

            # 计算边界框重叠情况
            for i in range(len(person_bboxes)):
                for j in range(i + 1, len(person_bboxes)):
                    if calculate_iou(person_bboxes[i], person_bboxes[j]) > 0:
                        bbox_overlap = 1
                        break
                if bbox_overlap:
                    break

        frame_idx += 1

    cap.release()

    total_persons = max(detected_persons) if detected_persons else 0
    return idx, 1 if total_persons > 0 else 0, total_persons, bbox_overlap  # 返回检测结果


def worker(task):
    """ 处理单个视频，分配 GPU """
    idx, video_path = task
    gpu_id = gpu_ids[idx % num_gpus]  # 轮流分配 GPU
    return process_video(idx, video_path, gpu_id)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # 多进程池，使用所有 GPU 进行并行推理
    pool = mp.Pool(processes=num_gpus)  # 进程数设为 GPU 数量，提高吞吐量

    # 并行处理视频
    results = list(tqdm(pool.imap(worker, enumerate(video_paths)), total=num_videos, desc="Processing Videos"))

    # 关闭进程池
    pool.close()
    pool.join()

    # 更新 CSV 文件
    for idx, has_person, num_persons, bbox_overlap in results:
        df.at[idx, "has_person"] = has_person
        df.at[idx, "num_persons"] = num_persons
        df.at[idx, "bbox_overlap"] = bbox_overlap  # 更新重叠信息

    # 保存修改后的 CSV
    df.to_csv(csv_path, index=False)
    print(f"检测完成，原 CSV 文件已更新: {csv_path}")
