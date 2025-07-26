# technical_indicators.py
import pandas as pd
import numpy as np

def calculate_ema(data, period):
    """计算指数移动平均线"""
    return data.ewm(span=period).mean()

def calculate_sma(data, period):
    """计算简单移动平均线"""
    return data.rolling(window=period).mean()

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

def calculate_roc(data, period=12):
    """计算变化率指标 (Rate of Change)"""
    return data.diff(period) / data.shift(period) * 100

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """计算随机指标 (Stochastic Oscillator)"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_obv(close, volume):
    """计算能量潮指标 (On-Balance Volume)"""
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def calculate_mfi(high, low, close, volume, period=14):
    """计算资金流量指标 (Money Flow Index)"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = pd.Series(0.0, index=typical_price.index, dtype=float)
    negative_flow = pd.Series(0.0, index=typical_price.index, dtype=float)
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.iloc[i] = money_flow.iloc[i]
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

def calculate_williams_r(high, low, close, period=14):
    """计算威廉指标 (Williams %R)"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_adx(high, low, close, period=14):
    """计算平均趋向指标 (Average Directional Index)"""
    # 计算真实波幅 (True Range)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算方向移动 (Directional Movement)
    dm_plus = pd.Series(0.0, index=high.index, dtype=float)
    dm_minus = pd.Series(0.0, index=high.index, dtype=float)
    
    for i in range(1, len(high)):
        high_diff = high.iloc[i] - high.iloc[i-1]
        low_diff = low.iloc[i-1] - low.iloc[i]
        
        if high_diff > low_diff and high_diff > 0:
            dm_plus.iloc[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            dm_minus.iloc[i] = low_diff
    
    # 计算平滑值
    tr_smooth = tr.rolling(window=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()
    
    # 计算方向指标
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    
    # 计算ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return adx, di_plus, di_minus

def calculate_kama(close, period=10, fast_ema=2, slow_ema=30):
    """计算考夫曼自适应移动平均线 (Kaufman Adaptive Moving Average)"""
    change = abs(close - close.shift(period))
    volatility = change.rolling(window=period).sum()
    er = change / volatility
    sc = (er * (fast_ema - slow_ema) + slow_ema) ** 2
    kama = pd.Series(close.iloc[0], index=close.index)
    
    for i in range(1, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
    
    return kama

def add_technical_indicators(df):
    """为DataFrame添加技术指标"""
    df = df.copy()
    
    # 基础移动平均线
    df['sma_20'] = calculate_sma(df['close'], 20)
    df['sma_50'] = calculate_sma(df['close'], 50)
    
    # 指数移动平均线
    df['ema_12'] = calculate_ema(df['close'], 12)
    df['ema_26'] = calculate_ema(df['close'], 26)
    
    # 相对强弱指数
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    # MACD指标
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    
    # 布林带
    df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'], df['bb_position'] = calculate_bollinger_bands(df['close'])
    
    # 变化率指标
    df['roc_12'] = calculate_roc(df['close'], 12)
    
    # 随机指标
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'])
    
    # 能量潮指标
    df['obv'] = calculate_obv(df['close'], df['volume'])
    
    # 资金流量指标
    df['mfi_14'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], 14)
    
    # 威廉指标
    df['williams_r'] = calculate_williams_r(df['high'], df['low'], df['close'], 14)
    
    # 平均趋向指标
    df['adx'], df['di_plus'], df['di_minus'] = calculate_adx(df['high'], df['low'], df['close'], 14)
    
    # 考夫曼自适应移动平均线
    df['kama'] = calculate_kama(df['close'])
    
    # K线结构特征
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    df['body_ratio'] = abs(df['body']) / (df['high'] - df['low'])
    
    # 价格比率特征
    df['close_open_ratio'] = df['close'] / df['open']
    df['high_low_ratio'] = df['high'] / df['low']
    
    # 填充NaN值
    df = df.bfill().ffill()
    
    return df 