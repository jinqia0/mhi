import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# ====== 配置路径 ======
csv_path = "./panda_10k_interaction_score.csv"  # 直接修改原 CSV
output_folder = "results"  # 结果保存文件夹
os.makedirs(output_folder, exist_ok=True)

# 读取 CSV
df = pd.read_csv(csv_path)

# 选择 True Positive 视频 (num_persons >= 2 且 interaction_score = 5)
true_positive_videos = df[(df["num_persons"] >= 2) & (df["interaction_score"] == 5)].copy()

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_distance(p1, p2):
    """ 计算两点的欧式距离 """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def detect_interaction(video_path, dist_threshold):
    """ 对视频进行交互检测 """
    cap = cv2.VideoCapture(video_path)
    interaction_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换颜色格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        persons = []  # 存储所有人的关键点
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                persons.append((x, y))

        # 检测多人交互
        if len(persons) >= 2:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    dist = calculate_distance(persons[i], persons[j])
                    if dist < dist_threshold:  # 可调整
                        interaction_detected = True
                        break

        if interaction_detected:
            break

    cap.release()
    return interaction_detected

# 遍历不同阈值
dist_thresholds = [10, 25, 50, 100]
interaction_results = {t: [] for t in dist_thresholds}

for idx, row in tqdm(true_positive_videos.iterrows(), total=len(true_positive_videos), desc="Processing Interaction Detection"):
    video_path = row["path"]

    # 检查视频是否存在
    if not os.path.exists(video_path):
        print(f"文件未找到: {video_path}")
        continue

    # 进行交互检测，分别记录不同阈值的结果
    for t in dist_thresholds:
        has_interaction = detect_interaction(video_path, t)
        interaction_results[t].append(has_interaction)

# 计算并保存不同阈值的比例
interaction_stats = []
for t in dist_thresholds:
    total_videos = len(interaction_results[t])
    detected_videos = sum(interaction_results[t])
    interaction_ratio = detected_videos / total_videos if total_videos > 0 else 0
    interaction_stats.append({"threshold": t, "interaction_ratio": interaction_ratio, "detected_videos": detected_videos, "total_videos": total_videos})

# 保存结果到 CSV
interaction_df = pd.DataFrame(interaction_stats)
interaction_df.to_csv("interaction_detection_results.csv", index=False)

print("交互检测完成，结果已保存至 interaction_detection_results.csv")
