import os
import logging
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import multiprocessing

# 修复1：强制设置多进程启动方式
multiprocessing.set_start_method('spawn', force=True)

SYSTEM_PROMPT = """Determine if the following video caption describes a scene where multiple people are engaged in **some form of** interaction. 
                The caption must explicitly describe people engaging in a physical or social interaction with another person, such as shaking hands, playing sports, or hugging. If the interaction is only implied or unclear, answer 'No'.
                You must **ONLY** respond with one of the following two words: **'Yes'** or **'No'**. 
                Do not add any explanation, punctuation, or extra words. Just output **one** of these two words."""

class InteractionDataset(Dataset):
    def __init__(self, df, tokenizer, system_prompt):
        self.df = df
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": row["caption"]},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False), idx

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 分布式初始化
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    
    # 修复2：统一数据广播机制
    obj_list = [None]  # 所有进程统一初始化
    
    if local_rank == 0:
        try:
            full_df = pd.read_csv("/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/panda/panda_10k_interaction_score.csv")
            full_df['caption_interaction'] = None
            logger.info(f"主节点加载数据完成，样本数: {len(full_df)}")
            obj_list[0] = full_df
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    # 广播数据
    dist.broadcast_object_list(obj_list, src=0)
    
    # 验证数据
    if obj_list[0] is None:
        raise RuntimeError("数据广播失败，接收到空数据")
    full_df = obj_list[0]

    # 初始化模型
    tokenizer = AutoTokenizer.from_pretrained("./LLM/Llama-3.1-8B-Instruct/")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 修复3：确保模型在主进程初始化后广播权重
    model = AutoModelForCausalLM.from_pretrained(
        "./LLM/Llama-3.1-8B-Instruct/",
        torch_dtype=torch.bfloat16
    ).to(local_rank)
    
    # 创建数据集
    dataset = InteractionDataset(full_df, tokenizer, SYSTEM_PROMPT)
    sampler = DistributedSampler(dataset, shuffle=False)
    
    # 修复4：调整数据加载配置
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,  # 暂时禁用多进程加载
        pin_memory=True,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            [item[1] for item in batch]
        )
    )

    # 推理循环
    results = {}
    with torch.inference_mode():
        for texts, indices in tqdm(dataloader, disable=(local_rank != 0)):
            # 修复5：在主进程执行设备移动
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # 解析结果
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for idx, text in zip(indices, decoded):
                results[idx] = "Yes" if "Yes" in text.split("assistant")[-1] else "No"

    # 收集结果
    all_results = [None] * world_size
    dist.gather_object(results, all_results if local_rank == 0 else None, dst=0)

    # 保存结果
    if local_rank == 0:
        final_df = full_df.copy()
        final_df['caption_interaction'] = final_df.index.map(
            {k:v for res in all_results if res is not None for k,v in res.items()}
        )
        final_df.to_csv("/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/panda/panda_10k_interaction_score_mp.csv", index=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"[RANK {os.environ.get('LOCAL_RANK', '?')}] 致命错误: {str(e)}", exc_info=True)
        dist.destroy_process_group()
        raise
