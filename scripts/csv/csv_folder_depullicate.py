import os
import glob
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# 输入文件夹路径，假设文件存储在 'input_folder' 中
input_folder = 'Datasets/mhi'
output_folder = 'Datasets/mhi_depulicated'
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有CSV文件
csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

# 遍历每个CSV文件并处理
for csv_file in csv_files:
    # 显式指定列的数据类型，防止 Dask 错误推断
    dtype = {
        'path': 'string',  # 将 path 列设置为字符串类型
        'caption': 'string',  # 将 caption 列设置为字符串类型
        'aes': 'float64',  # 将 aes 列设置为浮点数类型
        'video_id': 'string',  # 将 video_id 列设置为字符串类型
        'ocr': 'float64',  # 将 ocr 列设置为浮点数类型
        'flow': 'float64',  # 将 flow 列设置为浮点数
    }

    # 读取CSV文件，指定dtype
    df = dd.read_csv(csv_file, dtype=dtype)
    
    # 提取 filename 列
    df['filename'] = df['path'].apply(lambda x: os.path.basename(x), meta=('x', 'string'))
    
    # 基于 filename 列进行去重，保留所有列
    df_deduplicated = df.drop_duplicates(subset=['filename'])
    
    # 生成输出文件路径
    output_file = os.path.join(output_folder, os.path.basename(csv_file))
    
    # 保存去重后的文件，保留所有列
    df_deduplicated.to_csv(output_file, index=False, single_file=True)
    
    # 执行计算并保存进度
    with ProgressBar():
        df_deduplicated.compute()

    print(f'Processed and saved: {output_file}')