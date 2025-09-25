import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# 读取CSV文件，使用Dask DataFrame
df = dd.read_csv('file_scan_output/merged_file.csv', usecols=['filename', 'filepath'])

# 进行去重操作
total_rows_before = len(df)  # 计算去重前的行数
df_deduplicated = df.drop_duplicates(subset=['filename'])

# 计算去重后的行数
total_rows_after = len(df_deduplicated)

# 计算去重的行数
deduplicated_rows = total_rows_before - total_rows_after

# 保存结果到新的CSV文件
df_deduplicated.to_csv('deduplicated_merged_output.csv', index=False, single_file=True)

# 显示进度条并执行计算
with ProgressBar():
    df_deduplicated.compute()

# 输出去重前和去重后的行数，以及去重的行数
print(f'Total rows before deduplication: {total_rows_before}')
print(f'Total rows after deduplication: {total_rows_after}')
print(f'Total rows removed during deduplication: {deduplicated_rows}')
