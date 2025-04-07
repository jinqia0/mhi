import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    # 文件路径配置
    csv_path = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/internvid_rename_col.csv"

    # 读取CSV并生成path列
    df = pd.read_csv(csv_path)
    
    df = df[['path', 'caption', 'aes', 'video_id']]

    df.to_csv(csv_path, index=False)
    print("CSV文件已更新")
    