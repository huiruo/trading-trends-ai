# technical_indicators.py - 改进版技术指标计算
import pandas as pd
import numpy as np
from typing import List

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加改进的技术指标，使用更平稳的特征
    """
    df = df.copy()
    
    # ===== 基础平稳特征 =====
    # 1. 对数收益率 (平稳特征)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. 高低价比率 (平稳特征)
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    
    # 3. 成交量对数收益率 (平稳特征)
    df['volume_log_return'] = np.log(df['volume'] / df['volume'].shift(1))
    
    # 4. 价格在当日区间的位置 (平稳特征)
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # ===== 技术指标 =====
    # 5. RSI (14期)
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    
    # 6. 布林带位置
    df['bb_position'] = calculate_bollinger_position(df['close'], 20)
    
    # 7. MACD
    df['macd_histogram'] = calculate_macd_histogram(df['close'])
    
    # 8. 移动平均线交叉信号
    df['ma_cross_signal'] = calculate_ma_cross_signal(df['close'])
    
    # 9. 成交量与移动平均线比率
    df['volume_ma_ratio'] = calculate_volume_ma_ratio(df['volume'])
    
    # 10. 5日动量
    df['momentum_5'] = calculate_momentum(df['close'], 5)
    
    # ===== Z-score标准化 =====
    # 对需要标准化的指标进行z-score计算
    rolling_window = 100  # 滚动窗口大小
    
    # RSI的z-score
    df['rsi_14_zscore'] = calculate_zscore(df['rsi_14'], rolling_window)
    
    # MACD柱状图的z-score
    df['macd_histogram_zscore'] = calculate_zscore(df['macd_histogram'], rolling_window)
    
    # 动量的z-score
    df['momentum_5_zscore'] = calculate_zscore(df['momentum_5'], rolling_window)
    
    # 清理NaN值
    df = df.ffill().bfill()
    
    return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_position(prices: pd.Series, period: int = 20) -> pd.Series:
    """计算价格在布林带中的位置 (0-1范围)"""
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = ma + (2 * std)
    lower_band = ma - (2 * std)
    
    # 计算位置：0=下轨，0.5=中轨，1=上轨
    position = (prices - lower_band) / (upper_band - lower_band)
    return position.clip(0, 1)  # 限制在0-1范围内

def calculate_macd_histogram(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """计算MACD柱状图"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return histogram

def calculate_ma_cross_signal(prices: pd.Series, short_period: int = 5, long_period: int = 20) -> pd.Series:
    """计算均线交叉信号 (-1=死叉, 0=无信号, 1=金叉)"""
    ma_short = prices.rolling(window=short_period).mean()
    ma_long = prices.rolling(window=long_period).mean()
    
    # 计算交叉信号
    signal = pd.Series(0, index=prices.index)
    signal[(ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))] = 1   # 金叉
    signal[(ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))] = -1  # 死叉
    
    return signal

def calculate_volume_ma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """计算成交量与移动平均线比率"""
    volume_ma = volume.rolling(window=period).mean()
    ratio = volume / volume_ma
    return ratio

def calculate_momentum(prices: pd.Series, period: int = 5) -> pd.Series:
    """计算动量指标"""
    return prices / prices.shift(period) - 1

def calculate_zscore(series: pd.Series, window: int = 100) -> pd.Series:
    """计算滚动z-score标准化"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore

def get_feature_importance_analysis(df: pd.DataFrame) -> dict:
    """
    分析特征的重要性，帮助识别冗余特征
    """
    from config_improved import FEATURE_COLUMNS
    
    # 计算特征间的相关性
    feature_df = df[FEATURE_COLUMNS].copy()
    correlation_matrix = feature_df.corr()
    
    # 找出高相关性的特征对
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:  # 高相关性阈值
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
    
    # 计算每个特征的方差（低方差可能表示信息量少）
    feature_variance = feature_df.var().to_dict()
    
    return {
        'high_correlation_pairs': high_corr_pairs,
        'feature_variance': feature_variance,
        'total_features': len(FEATURE_COLUMNS)
    } 