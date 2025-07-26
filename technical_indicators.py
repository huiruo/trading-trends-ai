# technical_indicators.py
import pandas as pd
import numpy as np

def calculate_ema(data, period):
    """计算指数移动平均线"""
    return data.ewm(span=period).mean()

def calculate_rsi(data, period=14):
    """计算相对强弱指数"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """计算布林带"""
    bb_middle = data.rolling(window=period).mean()
    bb_std = data.rolling(window=period).std()
    bb_upper = bb_middle + (bb_std * std_dev)
    bb_lower = bb_middle - (bb_std * std_dev)
    bb_width = bb_upper - bb_lower
    bb_position = (data - bb_lower) / bb_width
    return bb_upper, bb_middle, bb_lower, bb_width, bb_position

def add_technical_indicators(df):
    """为DataFrame添加技术指标"""
    df = df.copy()
    
    # 计算EMA
    df['ema_12'] = calculate_ema(df['close'], 12)
    df['ema_26'] = calculate_ema(df['close'], 26)
    
    # 计算RSI
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    # 计算MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    
    # 计算布林带
    df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'], df['bb_position'] = calculate_bollinger_bands(df['close'])
    
    # 填充NaN值
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df 