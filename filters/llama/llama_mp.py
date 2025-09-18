import os
import torch
import torch.distributed as dist
import pandas as pd
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from tqdm import tqdm
import math

# 初始化分布式进程
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

# 设置当前 GPU（假设每个进程使用不同的 GPU）
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")

# 禁用 transformers 日志
logging.getLogger("transformers").setLevel(logging.ERROR)

# 加载模型和分词器（所有进程均加载）
model_path = "./LLM/Llama-3.1-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# --- 数据加载部分 ---
csv_file = "./data/panda_100k.csv"
if rank == 0:
    df = pd.read_csv(csv_file)  # 仅 rank 0 读取 CSV
    # 添加新列
    df['caption_interaction'] = None
    df_bytes = pickle.dumps(df)
else:
    df = None
    df_bytes = None

# 广播 DataFrame（转换为 bytes 后广播）
df_bytes_list = [df_bytes]
dist.broadcast_object_list(df_bytes_list, src=0)
# 反序列化得到 DataFrame
df = pickle.loads(df_bytes_list[0])
if rank == 0:
    print("DataFrame successfully broadcasted to all ranks.")

# --- 分片处理 --- 
total_rows = len(df)
# 计算每个进程处理的起始和结束索引
rows_per_rank = math.ceil(total_rows / world_size)
start_idx = rank * rows_per_rank
end_idx = min(start_idx + rows_per_rank, total_rows)
if rank == 0:
    print(f"Total rows: {total_rows}, rows per rank: {rows_per_rank}")

# 每个进程仅处理自己分片部分
df_slice = df.iloc[start_idx:end_idx].copy()
print(f"Rank {rank} processing rows {start_idx} to {end_idx}")

for index, row in tqdm(df_slice.iterrows(), total=len(df_slice), desc=f"Rank {rank} Processing"):
    caption = row["text"]

    messages = [
        {"role": "system", "content": """
            Determine if the following video caption describes a scene where multiple people are engaged in **some form of** interaction. 
            The caption must explicitly describe people engaging in a physical or social interaction with another person, such as shaking hands, playing sports, or hugging. If the interaction is only implied or unclear, answer 'No'.
            You must **ONLY** respond with one of the following two words: **'Yes'** or **'No'**. 
            Do not add any explanation, punctuation, or extra words. Just output **one** of these two words.
        """},
        {"role": "user", "content": caption},
    ]
    # 使用 transformer 提供的模板方法（确保该方法在当前 tokenizer 中可用）
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            top_p=0.9,
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    result = generated_text.split("assistant")[-1].strip()

    # 更新分片 DataFrame 中对应行的结果
    df_slice.at[index, 'caption_interaction'] = result

# 将每个进程的分片保存到本地临时文件（例如：temp_result_rank{rank}.pkl）
temp_filename = f"./data/temp_result_rank{rank}.pkl"
df_slice.to_pickle(temp_filename)
print(f"Rank {rank} saved its result to {temp_filename}")

# --- 汇总结果 --- 
# 等待所有进程完成
dist.barrier()

# 让 rank 0 汇总各个进程的结果
if rank == 0:
    df_list = [df_slice]  # rank 0 自己的结果已经在 df_slice 中
    for r in range(1, world_size):
        temp_file = f"./data/temp_result_rank{r}.pkl"
        # 等待文件存在（或者直接假定所有进程都已经保存完成）
        temp_df = pd.read_pickle(temp_file)
        df_list.append(temp_df)
    # 合并所有分片，注意合并时按原 DataFrame 行索引排序
    df_final = pd.concat(df_list).sort_index()
    csv_file_save = "./data/interaction_100k.csv"
    df_final.to_csv(csv_file_save, index=False)
    print(f"Results saved to {csv_file_save}")

# 同步并关闭分布式环境
dist.barrier()
dist.destroy_process_group()
