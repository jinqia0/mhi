import pandas as pd

# 读取 CSV
csv_path = "./panda_10k_interaction_score.csv"
df = pd.read_csv(csv_path)

# 真实的多人视频（标准1）：num_persons >= 2
true_multi_person_1 = df[df["num_persons"] >= 2]
total_true_multi_1 = len(true_multi_person_1)  # 真实的多人视频总数

# 真实的多人视频（标准2）：bbox_overlap = 1
true_multi_person_2 = df[df["bbox_overlap"] == 1]
total_true_multi_2 = len(true_multi_person_2)  # 真实的多人视频总数（包括重叠）

# 统计不同 interaction_score（1~5）下的正确率和漏检率
stats = []
detected_multi = df[df["caption_interaction"] == 'Yes']  # 预测为多人（interaction_score >= score）

# === 标准1：基于 num_persons 计算统计数据 ===
true_positives_1 = len(detected_multi[detected_multi["num_persons"] >= 2])  # 预测多人且真实为多人
predicted_positives_1 = len(detected_multi)  # 预测为多人的总数
false_negatives_1 = len(true_multi_person_1[true_multi_person_1["caption_interaction"] == 'No'])  # 真实多人但未预测多人

precision_1 = true_positives_1 / predicted_positives_1 if predicted_positives_1 > 0 else 0
miss_rate_1 = false_negatives_1 / total_true_multi_1 if total_true_multi_1 > 0 else 0

stats.append({
    "standard": "num_persons ≥ 2",
    "precision": round(precision_1, 4),
    "miss_rate": round(miss_rate_1, 4),
    "true_positives": true_positives_1,
    "false_negatives": false_negatives_1,
    "total_true_multi": total_true_multi_1
})

# === 标准2：基于 bbox_overlap 计算统计数据 ===
true_positives_2 = len(detected_multi[detected_multi["bbox_overlap"] == 1])
false_negatives_2 = len(true_multi_person_2[true_multi_person_2["caption_interaction"] == 'No'])  # 真实多人但未预测多人

precision_2 = true_positives_2 / predicted_positives_1 if predicted_positives_1 > 0 else 0  # 预测值不变
miss_rate_2 = false_negatives_2 / total_true_multi_2 if total_true_multi_2 > 0 else 0

stats.append({
    "standard": "bbox_overlap = 1",
    "precision": round(precision_2, 4),
    "miss_rate": round(miss_rate_2, 4),
    "true_positives": true_positives_2,
    "false_negatives": false_negatives_2,
    "total_true_multi": total_true_multi_2
})

# 转换为 DataFrame 并打印结果
stats_df = pd.DataFrame(stats)
print(stats_df)

# 保存结果
stats_df.to_csv("detection_accuracy_v2.csv", index=False)
print("检测正确率、漏检率和视频个数已保存至 detection_accuracy_v2.csv")
