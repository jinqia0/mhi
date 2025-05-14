import pandas as pd

def count_num_persons_distribution(csv_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 确保列存在且转换为数值类型
    df['num_persons'] = pd.to_numeric(df['num_persons'], errors='coerce')
    
    # 过滤无效值（非数值和空值）
    valid_data = df['num_persons'].dropna()
    
    # 计算各区间计数
    counts = {
        '0': ((valid_data == 0).sum()),
        '1': ((valid_data == 1).sum()),
        '2': ((valid_data == 2).sum()),
        '3': ((valid_data == 3).sum()),
        '4': ((valid_data == 4).sum()),
        '>4': ((valid_data > 4).sum())
    }
    
    # 计算总有效数
    total = valid_data.count()
    
    # 计算占比并格式化
    distribution = {
        key: round((value / total * 100), 2) 
        for key, value in counts.items()
    }
    
    return distribution

# 使用示例
result = count_num_persons_distribution("/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/internvid/slices/internvid_00-15_yolo.csv")
for k, v in result.items():
    print(f"{k}人: {v}%")
