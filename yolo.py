import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm  # 进度条显示

# ====== 配置路径 ======
csv_path = "./panda_10k_interaction_score.csv"   # 直接修改原 CSV
output_folder = "results"  # 结果保存文件夹
os.makedirs(output_folder, exist_ok=True)

# 1. 读取 CSV（仅取前1000行）
df = pd.read_csv(csv_path)
# df = df.head(1000)  # 仅处理前1000行

# 2. 加载 YOLOv8 预训练模型
model = YOLO("yolo11n.pt", verbose = False)  # 你也可以使用 'yolov8s.pt' 提高检测精度

# 3. 初始化新列（如果原 CSV 没有这些列，先填充 0）
if "has_person" not in df.columns:
    df["has_person"] = 0
if "num_persons" not in df.columns:
    df["num_persons"] = 0

# 4. 遍历所有视频进行人体检测
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
    video_path = row["path"]

    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"文件未找到: {video_path}")
        df.at[idx, "has_person"] = 0
        df.at[idx, "num_persons"] = 0
        continue

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设定检测间隔（确保长视频检测帧数多）
    step_size = max(1, total_frames // 30)
    frame_idx = 0
    detected_persons = set()  # 存储检测到的 "person" 数量

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 仅检测首帧 & 每 step_size 帧
        if frame_idx == 0 or frame_idx % step_size == 0:
            results = model.predict(frame,verbose=False)

            person_count = sum(1 for box in results[0].boxes if int(box.cls[0].item()) == 0)  # 统计当前帧人体数量

            if person_count > 0:
                detected_persons.add(person_count)

        frame_idx += 1

    cap.release()

    # 记录最终检测结果
    total_persons = max(detected_persons) if detected_persons else 0
    df.at[idx, "has_person"] = 1 if total_persons > 0 else 0
    df.at[idx, "num_persons"] = total_persons

# 5. **直接保存修改后的 CSV**
df.to_csv(csv_path, index=False)
print(f"检测完成，原 CSV 文件已更新: {csv_path}")
