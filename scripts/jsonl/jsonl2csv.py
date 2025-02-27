import json
import csv

# 文件路径
jsonl_file = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/intervid/internvid.aes18m.vc2vicuna.jsonl"
csv_file = "/mnt/pfs-gv8sxa/tts/dhg/jinqiao/mhi/Datasets/intervid/internvid.csv"

# 读取 JSONL 并写入 CSV
with open(jsonl_file, "r", encoding="utf-8") as infile, open(csv_file, "w", newline="", encoding="utf-8") as outfile:
    # 解析第一行以获取字段名
    first_line = json.loads(infile.readline())
    fieldnames = first_line.keys()
    
    # 回到文件开头
    infile.seek(0)

    # 写入 CSV
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for line in infile:
        writer.writerow(json.loads(line))

print(f"转换完成！CSV 文件已保存至：{csv_file}")
