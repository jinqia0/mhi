import easyocr  
import argparse  
from decord import VideoReader, cpu  
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

def process_video(path):  
    reader = easyocr.Reader(['ch_sim','en']) # create a reader object in each child process  
    # reader = easyocr.Reader(['ch_sim','en'], gpu=False) # if you don't want to use GPU
    video = VideoReader(path, ctx=cpu(0))    
    total_frames = len(video)    
    frames_to_extract = [total_frames * i // 6 for i in range(6)]    

    for frame_num in frames_to_extract:    
        frame = video[frame_num].asnumpy()    
        result = reader.readtext(frame, detail=0)    
        if result:    
            return False  
    return True  

def main(args):  
    df = pd.read_csv(args.csv_path)
    
    video_paths = df['path'].to_list()
    with Pool(args.num_workers) as p:
        results = list(tqdm(p.imap(process_video, video_paths), total=len(video_paths)))
    
    df['has_text'] = results
    
    rows = len(df)
    has_text = len(df['has_text'] == False)
    
    print(f"{args.csv_path} 处理完成：\n  总行数：{rows}\n  有文字的行数：{has_text}\n  占比：{round(100 * has_text / rows, 2) }%")

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()      
    parser.add_argument('-i', '--csv_path', help='Path to the input json file, Content is the path to the split video. If you do not have a json file, just pass the path to the video.')      
    parser.add_argument('--num_workers', type=int, default=32, help='Number of worker processes to use.')
    args = parser.parse_args()      
    main(args)