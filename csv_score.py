import pandas as pd

# 读取处理后的结果 CSV
input_csv_file = "./panda_10k_multi_level.csv"  # 你的输入文件
output_csv_file = "./panda_10k_interaction_score.csv"  # 处理后的输出文件

# 读取 CSV
df = pd.read_csv(input_csv_file)

# 定义评分计算函数
def get_interaction_score(row):
    if row["strict"] == "Yes":
        return 5
    elif row["standard"] == "Yes":
        return 4
    elif row["relaxed"] == "Yes":
        return 3
    elif row["lenient"] == "Yes":
        return 2
    elif row["very_lenient"] == "Yes":
        return 1
    else:
        return 0

# 计算 interaction_score
df["interaction_score"] = df.apply(get_interaction_score, axis=1)

# 删除原来的分类列
df = df.drop(columns=["strict", "standard", "relaxed", "lenient", "very_lenient"])

# 保存到新的 CSV 文件
df.to_csv(output_csv_file, index=False)
print(f"Results saved to {output_csv_file}")
