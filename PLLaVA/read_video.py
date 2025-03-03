# 可能的实现（基于 PyAV 库）
import av
import torch
import numpy as np

def read_video_av(video_path, pts_unit="sec", output_format="THWC"):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    frames = []
    for frame in container.decode(video_stream):
        img = frame.to_image()  # 转换为 PIL.Image
        tensor = torch.from_numpy(np.array(img))  # 转为 THWC 张量
        frames.append(tensor)
    vframes = torch.stack(frames)
    return vframes, None, {"fps": video_stream.average_rate}
