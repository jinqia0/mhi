import pandas as pd
import tqdm

csv_path = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/panda/panda_19-23_multi.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_path)

def update_path(path):
    # 替换路径中的特定部分
    video_name = path.split('/')[-1]
    new_path = f"/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Data/panda/part5/{video_name}"
    return new_path

tqdm.tqdm.pandas(desc="更新路径")
# 使用 tqdm 的 progress_apply 方法来显示进度条
df['path'] = df['path'].progress_apply(update_path)
# 保存更新后的 DataFrame 到新的 CSV 文件
output_file = csv_path # 可以更改为输出到其他目录或文件名
df.to_csv(output_file, index=False)
# 输出完成信息
print(f"过滤后的数据已保存为 {csv_path}")
