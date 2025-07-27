# technical_indicators.py
import pandas as pd
import numpy as np
from config_improved import FEATURE_COLUMNS

def calculate_rsi(df, period=14):
    """计算相对强弱指数"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """计算布林带"""
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    bb_upper = sma + (std * std_dev)
    bb_lower = sma - (std * std_dev)
    bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    return bb_position

def calculate_macd(df, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - signal_line
    return macd_histogram

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    low_min = df['low'].rolling(window=n).min()
    high_max = df['high'].rolling(window=n).max()
    
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=m1-1).mean()
    d = k.ewm(com=m2-1).mean()
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_moving_averages(df):
    """计算移动平均线"""
    ma5 = df['close'].rolling(window=5).mean()
    ma10 = df['close'].rolling(window=10).mean()
    ma20 = df['close'].rolling(window=20).mean()
    
    # 计算价格相对于移动平均线的位置
    ma5_ratio = df['close'] / ma5
    ma10_ratio = df['close'] / ma10
    ma20_ratio = df['close'] / ma20
    
    return ma5_ratio, ma10_ratio, ma20_ratio

def calculate_volume_indicators(df):
    """计算成交量指标"""
    # 成交量移动平均
    volume_ma5 = df['volume'].rolling(window=5).mean()
    volume_ma10 = df['volume'].rolling(window=10).mean()
    
    # 成交量比率
    volume_ratio_5 = df['volume'] / volume_ma5
    volume_ratio_10 = df['volume'] / volume_ma10
    
    # 价量关系
    price_volume_trend = (df['close'] - df['close'].shift(1)) * df['volume']
    pvt_ma = price_volume_trend.rolling(window=5).mean()
    
    return volume_ratio_5, volume_ratio_10, pvt_ma

def calculate_momentum_indicators(df):
    """计算动量指标"""
    # 价格动量
    momentum_5 = df['close'] / df['close'].shift(5) - 1
    momentum_10 = df['close'] / df['close'].shift(10) - 1
    
    # 价格变化率
    roc_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
    roc_10 = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    
    return momentum_5, momentum_10, roc_5, roc_10

def add_technical_indicators(df):
    """添加更多技术指标到DataFrame"""
    df = df.copy()
    
    # 修复FutureWarning
    df = df.bfill().ffill()
    
    # 基础指标
    if 'rsi_14' in FEATURE_COLUMNS:
        df['rsi_14'] = calculate_rsi(df, 14)
    
    if 'bb_position' in FEATURE_COLUMNS:
        df['bb_position'] = calculate_bollinger_bands(df)
    
    if 'close_open_ratio' in FEATURE_COLUMNS:
        df['close_open_ratio'] = df['close'] / df['open']
    
    # 新增指标
    if 'macd_histogram' in FEATURE_COLUMNS:
        df['macd_histogram'] = calculate_macd(df)
    
    if 'kdj_k' in FEATURE_COLUMNS:
        k, d, j = calculate_kdj(df)
        df['kdj_k'] = k
        df['kdj_d'] = d
        df['kdj_j'] = j
    
    if 'ma5_ratio' in FEATURE_COLUMNS:
        ma5_ratio, ma10_ratio, ma20_ratio = calculate_moving_averages(df)
        df['ma5_ratio'] = ma5_ratio
        df['ma10_ratio'] = ma10_ratio
        df['ma20_ratio'] = ma20_ratio
    
    if 'volume_ratio_5' in FEATURE_COLUMNS:
        volume_ratio_5, volume_ratio_10, pvt_ma = calculate_volume_indicators(df)
        df['volume_ratio_5'] = volume_ratio_5
        df['volume_ratio_10'] = volume_ratio_10
        df['pvt_ma'] = pvt_ma
    
    if 'momentum_5' in FEATURE_COLUMNS:
        momentum_5, momentum_10, roc_5, roc_10 = calculate_momentum_indicators(df)
        df['momentum_5'] = momentum_5
        df['momentum_10'] = momentum_10
        df['roc_5'] = roc_5
        df['roc_10'] = roc_10
    
    # 处理NaN值
    df = df.bfill().ffill()
    
    return df 