# preprocess.py
# 用于读取 CSV、标准化特征、构造时间窗口等处理：
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
from config import WINDOW_SIZE, FEATURE_COLUMNS, TARGET_COLUMN

# 全局scaler，用于保存和加载
scaler = MinMaxScaler()
SCALER_PATH = "model/scaler.pkl"

def load_and_preprocess(df_or_path):
    """
    加载并预处理数据
    参数可以是DataFrame或文件路径
    """
    if isinstance(df_or_path, str):
        # 如果是文件路径，先加载数据
        df = load_klines_from_csv(df_or_path)
    else:
        df = df_or_path.copy()
    
    # 标准化特征
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
    
    # 保存scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    return df

def load_scaler():
    """加载已保存的scaler"""
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def create_sequences(df: pd.DataFrame, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(df) - window_size):
        x_seq = df[FEATURE_COLUMNS].iloc[i:i+window_size].values
        y_target = df[TARGET_COLUMN].iloc[i + window_size]
        X.append(x_seq)
        y.append(y_target)
    return np.array(X), np.array(y)

def inverse_transform_close(value: float):
    """反向转换收盘价"""
    loaded_scaler = load_scaler()
    if loaded_scaler is None:
        return value
    
    # 创建一个dummy数组，只设置close列的值
    dummy = np.zeros((1, len(FEATURE_COLUMNS)))
    close_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    dummy[0, close_idx] = value
    
    # 反向转换
    return loaded_scaler.inverse_transform(dummy)[0, close_idx]

# 使用示例: df = load_klines_from_csv("dataset/btc_1h.csv")
# print(df.head())
def load_klines_from_csv(filepath: str) -> pd.DataFrame:
    """
    加载币安格式的CSV K线数据，并标准化字段为 timestamp, open, high, low, close, volume
    """
    df = pd.read_csv(filepath)

    # 重命名为标准字段
    df = df.rename(columns={
        'closeTime': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

    # 保留需要的列
    df = df[['_id', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # 转换时间戳（从毫秒整型）
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['_id'] = pd.to_datetime(df['_id'], unit='ms')

    # 按_id字段（时间戳）升序排列，确保时间序列正确
    df = df.sort_values(by='_id')
    
    # 删除_id列，只保留timestamp
    df = df.drop('_id', axis=1)

    return df
