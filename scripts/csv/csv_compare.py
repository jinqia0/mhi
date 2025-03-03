import pandas as pd

# 读取两个CSV文件
file1 = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/panda/panda_10k_interaction_score_mp.csv'
file2 = '/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/data/panda/panda_10k_interaction_score.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 检查caption_interaction列是否相同
if 'caption_interaction' in df1.columns and 'caption_interaction' in df2.columns:
    if df1['caption_interaction'].equals(df2['caption_interaction']):
        print("caption_interaction 列完全相同")
    else:
        # 找出不同的行
        diff = df1[df1['caption_interaction'] != df2['caption_interaction']]
        print(f"caption_interaction 列不同，共有 {len(diff)} 行不同：")
        print(diff)
else:
    print("其中一个或两个文件中缺少 caption_interaction 列")
