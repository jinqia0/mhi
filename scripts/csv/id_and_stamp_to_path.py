import pandas as pd
from tqdm import tqdm

def id_and_stamp_to_path(row):
    # 提取id和stamp
    id = row['YoutubeID']
    start = row['Start_timestamp']
    end = row['End_timestamp']

    # 构建新的path
    new_path = f"{id}-{start}-{end}.mp4"

    # 返回新的path
    return new_path


if __name__ == '__main__':
    # 文件路径配置
    input_csv = "Datasets/internvid.csv"
    output_csv = "Datasets/internvid_rename_col.csv"

    # 读取CSV并生成path列
    df = pd.read_csv(input_csv)
    df_new = pd.DataFrame()
    
    tqdm.pandas()
    df_new['video_id'] = df['YoutubeID']
    df_new['path'] = df.progress_apply(id_and_stamp_to_path, axis=1)
    df_new['caption'] = df['Caption']
    df_new['aes'] = df['Aesthetic_Score']

    df_new.to_csv(output_csv, index=False)
    print("CSV文件已生成")
    