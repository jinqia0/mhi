import pandas as pd

def stat_ocr_distribution(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 假设 OCR 列为 'ocr'
    # 定义区间 0-1, 1-2, ..., 14-15
    bins = list(range(16))  # 生成 0 到 15 的边界
    labels = [f'{i}-{i+1}' for i in range(15)]  # 生成标签 0-1, 1-2, ..., 14-15

    # 使用 pd.cut() 分组
    df['ocr_group'] = pd.cut(df['ocr'], bins=bins, labels=labels, right=False)

    # 统计每组的个数
    group_counts = df['ocr_group'].value_counts().sort_index()

    # 打印结果
    print("OCR 分布统计:")
    print(group_counts)

    return group_counts

# 运行该函数，传入CSV文件路径
csv_path = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/mhi_multi_aes45_coarse_ocr.csv'  # 替换为你的CSV文件路径
stat_ocr_distribution(csv_path)
