import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm

# 禁用 transformers 的日志信息
logging.getLogger("transformers").setLevel(logging.ERROR)

# 加载模型和分词器
model_path = "./Meta-Llama-3.1-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).half()  # 混合精度加速
model = torch.nn.DataParallel(model)  # 多 GPU 处理
model.to("cuda")

# 读取 CSV 文件
csv_file = "./panda_10k_interaction_score.csv"  # 替换为你的文件路径
df = pd.read_csv(csv_file)

# 初始化新列
df['caption_interaction'] = None

# 设置批处理大小（根据 GPU 内存调整）
BATCH_SIZE = 4  

# 自定义数据集
class CaptionDataset(Dataset):
    def __init__(self, df):
        self.captions = df["caption"].tolist()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        messages = [
            {"role": "system", "content": """
                Determine if the following video caption describes a scene where multiple people are engaged in **some form of** interaction. 
                The caption must explicitly describe people engaging in a physical or social interaction with another person, such as shaking hands, playing sports, or hugging. If the interaction is only implied or unclear, answer 'No'.
                You must **ONLY** respond with one of the following two words: **'Yes'** or **'No'**. 
                Do not add any explanation, punctuation, or extra words. Just output **one** of these two words.
            """},
            {"role": "user", "content": caption},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return inputs, caption

# 创建数据加载器
dataset = CaptionDataset(df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 运行推理
results = []
model.eval()
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing captions"):
        inputs, captions = batch
        inputs = {key: val.squeeze(1).to("cuda") for key, val in inputs.items()}  # 送入 GPU

        with torch.cuda.amp.autocast():  # 混合精度加速
            output = model.module.generate(
                **inputs,
                max_new_tokens=10,  # 限制生成的长度
                temperature=0.1,    # 降低随机性，提高确定性
                top_p=0.9,          # 控制生成多样性
            )

        generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

        # 提取 assistant 的输出
        for caption, gen_text in zip(captions, generated_texts):
            result = gen_text.split("assistant")[-1].strip()
            results.append(result)

# 保存结果回 DataFrame
df['caption_interaction'] = results
df.to_csv(csv_file, index=False)
print(f"Results saved to {csv_file}")
