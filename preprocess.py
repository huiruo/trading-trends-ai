# preprocess.py
# 用于读取 CSV、标准化特征、构造时间窗口等处理：
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import pickle
import os
from config import WINDOW_SIZE, FEATURE_COLUMNS, TARGET_COLUMN
# 新增：如果有技术指标，优先用改进版配置
try:
    from config_improved import FEATURE_COLUMNS as IMPROVED_FEATURE_COLUMNS
    FEATURE_COLUMNS = IMPROVED_FEATURE_COLUMNS
except ImportError:
    pass
# 新增：引入技术指标
try:
    from technical_indicators import add_technical_indicators
except ImportError:
    def add_technical_indicators(df):
        return df

# 全局scaler，用于保存和加载
scaler = RobustScaler()
SCALER_PATH = "model/scaler.pkl"

def load_and_preprocess(df_or_path):
    """
    加载并预处理数据，支持DataFrame或文件路径。
    对所有特征进行标准化，使用更稳定的缩放方法。
    """
    if isinstance(df_or_path, str):
        df = load_klines_from_csv(df_or_path)
    else:
        df = df_or_path.copy()
    # 先加技术指标
    df = add_technical_indicators(df)
    
    # 对所有特征进行标准化
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
    
    # 检查是否使用分类方法
    try:
        from config_improved import USE_CLASSIFICATION, CLASSIFICATION_THRESHOLD
    except ImportError:
        USE_CLASSIFICATION = False
        CLASSIFICATION_THRESHOLD = 0.001
    
    for i in range(len(df) - window_size):
        x_seq = df[FEATURE_COLUMNS].iloc[i:i+window_size].values
        
        if USE_CLASSIFICATION:
            # 使用分类方法
            current_close = df[TARGET_COLUMN].iloc[i + window_size - 1]  # 当前收盘价
            next_close = df[TARGET_COLUMN].iloc[i + window_size]         # 下一个收盘价
            
            # 计算变化率
            if current_close == 0:
                change_ratio = 0
            else:
                change_ratio = (next_close - current_close) / current_close
            
            # 分类标签：0=跌，1=平，2=涨
            if change_ratio < -CLASSIFICATION_THRESHOLD:
                y_target = 0  # 跌
            elif change_ratio > CLASSIFICATION_THRESHOLD:
                y_target = 2  # 涨
            else:
                y_target = 1  # 平
        else:
            # 使用回归方法
            try:
                from config_improved import USE_RELATIVE_CHANGE, MAX_CHANGE_RATIO
            except ImportError:
                USE_RELATIVE_CHANGE = False
                MAX_CHANGE_RATIO = 0.05
            
            if USE_RELATIVE_CHANGE:
                # 使用相对变化作为目标
                current_close = df[TARGET_COLUMN].iloc[i + window_size - 1]  # 当前收盘价
                next_close = df[TARGET_COLUMN].iloc[i + window_size]         # 下一个收盘价
                
                # 防止除零错误
                if current_close == 0:
                    change_ratio = 0
                else:
                    change_ratio = (next_close - current_close) / current_close
                
                # 限制变化幅度在合理范围内
                change_ratio = max(-MAX_CHANGE_RATIO, min(MAX_CHANGE_RATIO, change_ratio))
                
                # 归一化到0-1范围
                y_target = (change_ratio + MAX_CHANGE_RATIO) / (2 * MAX_CHANGE_RATIO)
            else:
                # 使用绝对价格作为目标 - 需要反向转换到原始价格
                scaled_next_close = df[TARGET_COLUMN].iloc[i + window_size]
                # 反向转换到原始价格
                y_target = inverse_transform_close(scaled_next_close)
        
        X.append(x_seq)
        y.append(y_target)
    
    return np.array(X), np.array(y)

def inverse_transform_close(value: float):
    """反向转换收盘价"""
    loaded_scaler = load_scaler()
    if loaded_scaler is None:
        return value
    
    # 直接使用RobustScaler的反向转换公式
    # RobustScaler: (X - center_) / scale_
    # 反向转换: X = value * scale_ + center_
    close_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)
    original_value = value * loaded_scaler.scale_[close_idx] + loaded_scaler.center_[close_idx]
    
    return original_value

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
