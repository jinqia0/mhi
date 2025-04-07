import pandas as pd
from tqdm import tqdm

def path_to_id(path):
    # 返回新的path
    return path.split('/')[-1][:11]


if __name__ == '__main__':
    # 文件路径配置
    csv_path = "Datasets/panda_all_text_aes.csv"

    # 读取CSV并生成path列
    df = pd.read_csv(csv_path)
    
    tqdm.pandas()
    df['video_id'] = df.progress_apply(path_to_id, axis=1)
    df.rename(columns={'caption': 'text'}, inplace=True)
    df = df[['path', 'caption', 'aes', 'video_id']]

    df.to_csv(csv_path, index=False)
    print("CSV文件已更新")
    