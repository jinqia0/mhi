import torch
import torch.distributed as dist
import pandas as pd
import pickle
import math
import os

# 初始化分布式进程
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 设置当前 GPU（每个进程使用不同 GPU）
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")

# 数据加载：只有 rank 0 读取 CSV，其他进程等待广播
csv_file = "./data/panda_100k.csv"
if rank == 0:
    df = pd.read_csv(csv_file)
    # 为测试用，给 DataFrame 添加一列
    df['test_result'] = None
    df_bytes = pickle.dumps(df)
else:
    df = None
    df_bytes = None

# 广播 DataFrame（序列化为 bytes）
df_bytes_list = [df_bytes]
dist.broadcast_object_list(df_bytes_list, src=0)
df = pickle.loads(df_bytes_list[0])

# 分片处理：计算每个进程应处理的数据范围
total_rows = len(df)
rows_per_rank = int((total_rows + world_size - 1) / world_size)  # 向上取整
start_idx = rank * rows_per_rank
end_idx = min(start_idx + rows_per_rank, total_rows)

# 测试任务：每个进程在自己的分片中写入自己的 rank 号
df_slice = df.iloc[start_idx:end_idx].copy()
df_slice.loc[:, 'test_result'] = f"Processed by rank {rank}"

print(f"Rank {rank} processing rows {start_idx} to {end_idx}")

# 每个进程将自己的结果保存为临时文件
temp_filename = f"./data/temp_result_rank{rank}.pkl"
df_slice.to_pickle(temp_filename)
print(f"Rank {rank} saved its result to {temp_filename}")

# 同步所有进程
dist.barrier()

# 由 rank 0 汇总所有结果
if rank == 0:
    df_list = [pd.read_pickle(f"./data/temp_result_rank{r}.pkl") for r in range(world_size)]
    df_final = pd.concat(df_list).sort_index()
    csv_file_save = "./data/test_distributed.csv"
    df_final.to_csv(csv_file_save, index=False)
    print(f"Final results saved to {csv_file_save}")

dist.barrier()
dist.destroy_process_group()
