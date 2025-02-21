import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from tqdm import tqdm

# 禁用 transformers 的日志信息
logging.getLogger("transformers").setLevel(logging.ERROR)

device = "cuda:7"

# 加载模型和分词器
model_path = "./LLM/Llama-3.1-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# 读取 CSV 文件
csv_file = "./data/panda_100k.csv"  # 替换为你的文件路径
df = pd.read_csv(csv_file)

# Initialize the new column with None
df['caption_interaction'] = None

# 处理每个 caption 并进行判定
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing captions"):
    caption = row["caption"]
    
    messages = [
        {"role": "system", "content": f"""
            Determine if the following video caption describes a scene where multiple people are engaged in **some form of** interaction. 
            The caption must explicitly describe people engaging in a physical or social interaction with another person, such as shaking hands, playing sports, or hugging. If the interaction is only implied or unclear, answer 'No'.
            You must **ONLY** respond with one of the following two words: **'Yes'** or **'No'**. 
            Do not add any explanation, punctuation, or extra words. Just output **one** of these two words.
        """},
        {"role": "user", "content": caption},
    ]

    # 格式化输入
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 生成输出
    output = model.generate(
        **inputs,
        max_new_tokens=10,  # 限制生成的长度
        temperature=0.1,    # 降低随机性，提高确定性
        top_p=0.9,          # 控制生成多样性
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 提取 assistant 的输出
    result = generated_text.split("assistant")[-1].strip()

    # 保存结果到 DataFrame
    df.at[index, 'caption_interaction'] = result

# 将结果保存回 CSV 文件
csv_file_save = "./data/interaction_100k.csv"
df.to_csv(csv_file_save, index=False)
print(f"Results saved to {csv_file}")
