# 调试数据预处理问题
# python script_debug_data.py
import pandas as pd
import numpy as np
from preprocess import load_and_preprocess, create_sequences
from technical_indicators import add_technical_indicators
from config_improved import *

def debug_data():
    print("=== 数据调试 ===")
    
    # 1. 加载原始数据
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        'closeTime': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    print(f"1. 原始数据范围:")
    print(f"   close: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"   open: {df['open'].min():.2f} - {df['open'].max():.2f}")
    print(f"   volume: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
    
    # 2. 添加技术指标
    df_with_indicators = add_technical_indicators(df)
    
    print(f"\n2. 添加技术指标后:")
    for col in ['rsi_14', 'bb_position', 'close_open_ratio']:
        if col in df_with_indicators.columns:
            values = df_with_indicators[col].dropna()
            print(f"   {col}: {values.min():.6f} - {values.max():.6f}")
    
    # 3. 检查是否有异常值
    print(f"\n3. 异常值检查:")
    for col in FEATURE_COLUMNS:
        if col in df_with_indicators.columns:
            values = df_with_indicators[col]
            inf_count = np.isinf(values).sum()
            nan_count = values.isna().sum()
            print(f"   {col}: inf={inf_count}, nan={nan_count}")
    
    # 4. 使用预处理函数
    df_processed = load_and_preprocess(DATA_PATH)
    
    print(f"\n4. 预处理后数据范围:")
    for col in FEATURE_COLUMNS:
        if col in df_processed.columns:
            values = df_processed[col]
            print(f"   {col}: {values.min():.6f} - {values.max():.6f}")
    
    # 5. 创建序列
    X, y = create_sequences(df_processed, window_size=WINDOW_SIZE)
    
    print(f"\n5. 序列数据:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   X range: {X.min():.6f} - {X.max():.6f}")
    print(f"   y range: {y.min():.6f} - {y.max():.6f}")
    
    # 6. 检查最后几个样本
    print(f"\n6. 最后3个样本的X值:")
    for i in range(max(0, len(X)-3), len(X)):
        print(f"   样本{i}: X范围={X[i].min():.6f} - {X[i].max():.6f}, y={y[i]:.6f}")

if __name__ == "__main__":
    debug_data() 