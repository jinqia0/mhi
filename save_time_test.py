import os
import time

# 文件路径
file_path = "save_time_test.txt"  # 请根据实际情况修改路径

# 创建并打开文件进行写入
with open(file_path, "w") as file:
    # 初始写入内容
    file.write("Initial content.\n")
    file.flush()
    
    # 获取文件的初始修改时间
    last_modified_time = os.path.getmtime(file_path)
    print(f"Initial write at: {time.ctime(last_modified_time)}")

    # 反复写入文件，测试文件更新时间
    for i in range(5):
        time.sleep(2)  # 每2秒写入一次
        file.write(f"Appending content {i + 1}.\n")
        file.flush()  # 刷新到文件
        last_modified_time = os.path.getmtime(file_path)
        print(f"Write {i + 1} at: {time.ctime(last_modified_time)}")
        
    print("Test completed.")
