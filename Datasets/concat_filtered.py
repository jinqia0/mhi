import pandas as pd
import glob
import os

# ✅ 设置目标文件夹路径
data_dir = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/panda"
output_file = os.path.join(data_dir, "panda_multi_aes45_coarse.csv")

# 获取所有以 part1~part4.csv 结尾的文件
pattern = os.path.join(data_dir, "*part[1-3].csv")
target_files = glob.glob(pattern)

dfs = []
for file in target_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # 清理列名空格

    if 'num_main_persons' in df.columns:
        df_filtered = df[df['num_main_persons'] >= 2]
        dfs.append(df_filtered)
    else:
        print(f"⚠️  警告：文件 {os.path.basename(file)} 缺少 'num_main_persons' 列，跳过。")

# 合并结果并保存
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"✅ 合并完成：共 {len(combined_df)} 行，已保存为：{output_file}")
else:
    print("❌ 没有找到任何包含 'num_main_persons >= 2' 的数据。")
