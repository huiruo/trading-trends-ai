import pandas as pd
from preprocess import load_klines_from_csv

# python -m test_sort.py
# 测试数据排序
from config_improved import DATA_PATH
df = load_klines_from_csv(DATA_PATH)

print("数据排序验证:")
print(f"总数据条数: {len(df)}")
print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
print("\n前5条数据:")
print(df.head()[['timestamp', 'open', 'close']])
print("\n后5条数据:")
print(df.tail()[['timestamp', 'open', 'close']])

# 检查时间间隔是否一致
time_diffs = df['timestamp'].diff().dropna()
print(f"\n时间间隔统计:")
print(f"平均间隔: {time_diffs.mean()}")
print(f"最小间隔: {time_diffs.min()}")
print(f"最大间隔: {time_diffs.max()}")

# 检查是否有重复时间
duplicates = df['timestamp'].duplicated().sum()
print(f"\n重复时间戳数量: {duplicates}") 

