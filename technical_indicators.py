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

def add_technical_indicators(df):
    """添加核心技术指标到DataFrame"""
    df = df.copy()
    
    # 修复FutureWarning
    df = df.bfill().ffill()
    
    # 只计算最核心的3个指标
    if 'rsi_14' in FEATURE_COLUMNS:
        df['rsi_14'] = calculate_rsi(df, 14)
    
    if 'bb_position' in FEATURE_COLUMNS:
        df['bb_position'] = calculate_bollinger_bands(df)
    
    if 'close_open_ratio' in FEATURE_COLUMNS:
        df['close_open_ratio'] = df['close'] / df['open']
    
    # 处理NaN值
    df = df.bfill().ffill()
    
    return df 