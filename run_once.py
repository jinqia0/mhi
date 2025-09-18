import torch
print(torch.cuda.is_available())  # 检查是否有可用的 GPU
# print(torch.cuda.device_count())  # 获取可用 GPU 数量
# print(torch.cuda.current_device())  # 当前设备
# print(torch.cuda.get_device_name(0))  # 显示第 0 个设备的名称
