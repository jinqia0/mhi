import pandas as pd

# 输入输出路径
input_csv = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/mhi_multi_aes45_coarse.csv"
output_csv = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/mhi_multi_aes45_coarse_merge.csv"

# 读取 CSV
df = pd.read_csv(input_csv)

# 如果同时存在 'text' 和 'caption' 列，检查是否都非空
if 'text' in df.columns and 'caption' in df.columns:
    # 找到同时非空的行
    conflict_mask = df['text'].notna() & df['caption'].notna()
    if conflict_mask.any():
        conflict_rows = df[conflict_mask]
        raise ValueError(f"存在 {conflict_rows.shape[0]} 行 'text' 和 'caption' 同时非空，程序中止。")

    # 否则用 text 补充 caption
    df['caption'] = df['caption'].fillna(df['text'])
    df.drop(columns=['text'], inplace=True)

# 仅存在 text 列，直接重命名为 caption
elif 'text' in df.columns:
    df.rename(columns={'text': 'caption'}, inplace=True)

# 保存修改后的文件
df.to_csv(output_csv, index=False)
print(f"✅ 已保存修改后的文件到: {output_csv}")
