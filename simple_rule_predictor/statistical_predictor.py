# statistical_predictor.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from technical_indicators import add_technical_indicators
from config_improved import DATA_PATH

def calculate_statistical_prediction(df, silent=False):
    """统计预测器 - 基于均值回归和统计套利"""
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 获取最新的技术指标
    latest = df.iloc[-1]
    
    # 1. 价格均值回归分析
    mean_reversion_signal = analyze_mean_reversion(df)
    
    # 2. 波动率分析
    volatility_signal = analyze_volatility_regime(df)
    
    # 3. 动量分析
    momentum_signal = analyze_momentum(df)
    
    # 4. 成交量分析
    volume_signal = analyze_volume_pattern(df)
    
    # 5. 技术指标背离分析
    divergence_signal = analyze_divergence(df)
    
    # 6. 综合决策
    final_decision = make_statistical_decision(
        df, mean_reversion_signal, volatility_signal, 
        momentum_signal, volume_signal, divergence_signal
    )
    
    # 只在非静默模式下输出预测方向
    if not silent:
        pred_change_ratio_pct = final_decision['涨跌幅度'].replace('%', '')
        print(f"🔍 预测方向: {final_decision['预测涨跌']}, 变化幅度: {pred_change_ratio_pct}%")
    
    return final_decision

def analyze_mean_reversion(df):
    """分析价格均值回归"""
    last_50_closes = df['close'].tail(50).values
    last_20_closes = df['close'].tail(20).values
    last_10_closes = df['close'].tail(10).values
    
    if len(last_50_closes) < 50:
        return 0
    
    # 计算多个时间框架的均值
    mean_50 = np.mean(last_50_closes)
    mean_20 = np.mean(last_20_closes)
    mean_10 = np.mean(last_10_closes)
    
    current_price = last_10_closes[-1]
    
    # 计算偏离度
    deviation_50 = (current_price - mean_50) / mean_50
    deviation_20 = (current_price - mean_20) / mean_20
    deviation_10 = (current_price - mean_10) / mean_10
    
    # 综合偏离度
    avg_deviation = (deviation_50 + deviation_20 + deviation_10) / 3
    
    # 均值回归信号
    if avg_deviation > 0.02:  # 价格高于均值2%以上
        return -2  # 强烈看跌（回归均值）
    elif avg_deviation > 0.01:  # 价格高于均值1%以上
        return -1  # 温和看跌
    elif avg_deviation < -0.02:  # 价格低于均值2%以上
        return 2   # 强烈看涨（回归均值）
    elif avg_deviation < -0.01:  # 价格低于均值1%以上
        return 1   # 温和看涨
    else:
        return 0   # 接近均值

def analyze_volatility_regime(df):
    """分析波动率状态"""
    last_20_closes = df['close'].tail(20).values
    
    if len(last_20_closes) < 20:
        return 0
    
    # 计算滚动波动率
    returns = np.diff(last_20_closes) / last_20_closes[:-1]
    volatility = np.std(returns)
    
    # 计算历史波动率分位数
    all_returns = df['close'].pct_change().dropna()
    if len(all_returns) > 100:
        vol_percentile = np.percentile(all_returns.rolling(20).std().dropna(), 80)
        
        if volatility > vol_percentile:
            return 0  # 高波动率时信号减弱
        elif volatility < np.percentile(all_returns.rolling(20).std().dropna(), 20):
            return 1  # 低波动率时信号增强
        else:
            return 0.5  # 正常波动率
    else:
        return 0.5

def analyze_momentum(df):
    """分析价格动量"""
    last_20_closes = df['close'].tail(20).values
    
    if len(last_20_closes) < 20:
        return 0
    
    # 计算多个时间框架的动量
    momentum_5 = (last_20_closes[-1] - last_20_closes[-5]) / last_20_closes[-5]
    momentum_10 = (last_20_closes[-1] - last_20_closes[-10]) / last_20_closes[-10]
    momentum_20 = (last_20_closes[-1] - last_20_closes[0]) / last_20_closes[0]
    
    # 动量一致性
    if momentum_5 > 0 and momentum_10 > 0 and momentum_20 > 0:
        return 2  # 强势上涨
    elif momentum_5 > 0 and momentum_10 > 0:
        return 1  # 温和上涨
    elif momentum_5 < 0 and momentum_10 < 0 and momentum_20 < 0:
        return -2  # 强势下跌
    elif momentum_5 < 0 and momentum_10 < 0:
        return -1  # 温和下跌
    else:
        return 0  # 动量不一致

def analyze_volume_pattern(df):
    """分析成交量模式"""
    last_10_volumes = df['volume'].tail(10).values
    last_10_closes = df['close'].tail(10).values
    
    if len(last_10_volumes) < 10:
        return 0
    
    # 计算成交量趋势
    volume_trend = (last_10_volumes[-1] - last_10_volumes[0]) / last_10_volumes[0]
    
    # 计算价格趋势
    price_trend = (last_10_closes[-1] - last_10_closes[0]) / last_10_closes[0]
    
    # 价量配合分析
    if price_trend > 0.01 and volume_trend > 0.2:  # 价涨量增
        return 2
    elif price_trend > 0.005 and volume_trend > 0.1:  # 价涨量增
        return 1
    elif price_trend < -0.01 and volume_trend > 0.2:  # 价跌量增
        return -2
    elif price_trend < -0.005 and volume_trend > 0.1:  # 价跌量增
        return -1
    elif price_trend > 0.005 and volume_trend < -0.1:  # 价涨量缩
        return -1  # 背离信号
    elif price_trend < -0.005 and volume_trend < -0.1:  # 价跌量缩
        return 1   # 背离信号
    else:
        return 0

def analyze_divergence(df):
    """分析技术指标背离"""
    if len(df) < 20:
        return 0
    
    # RSI背离分析
    rsi_divergence = analyze_rsi_divergence(df)
    
    # MACD背离分析
    macd_divergence = analyze_macd_divergence(df)
    
    return rsi_divergence + macd_divergence

def analyze_rsi_divergence(df):
    """分析RSI背离"""
    last_20 = df.tail(20)
    
    # 寻找价格和RSI的极值点
    price_highs = []
    price_lows = []
    rsi_highs = []
    rsi_lows = []
    
    for i in range(1, len(last_20) - 1):
        # 价格高点
        if last_20['close'].iloc[i] > last_20['close'].iloc[i-1] and last_20['close'].iloc[i] > last_20['close'].iloc[i+1]:
            price_highs.append((i, last_20['close'].iloc[i]))
            rsi_highs.append((i, last_20['rsi_14'].iloc[i]))
        
        # 价格低点
        if last_20['close'].iloc[i] < last_20['close'].iloc[i-1] and last_20['close'].iloc[i] < last_20['close'].iloc[i+1]:
            price_lows.append((i, last_20['close'].iloc[i]))
            rsi_lows.append((i, last_20['rsi_14'].iloc[i]))
    
    # 分析背离
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        # 顶背离：价格创新高，RSI未创新高
        if price_highs[-1][1] > price_highs[-2][1] and rsi_highs[-1][1] < rsi_highs[-2][1]:
            return -1  # 看跌信号
    
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        # 底背离：价格创新低，RSI未创新低
        if price_lows[-1][1] < price_lows[-2][1] and rsi_lows[-1][1] > rsi_lows[-2][1]:
            return 1   # 看涨信号
    
    return 0

def analyze_macd_divergence(df):
    """分析MACD背离"""
    last_20 = df.tail(20)
    
    # 寻找价格和MACD的极值点
    price_highs = []
    price_lows = []
    macd_highs = []
    macd_lows = []
    
    for i in range(1, len(last_20) - 1):
        # 价格高点
        if last_20['close'].iloc[i] > last_20['close'].iloc[i-1] and last_20['close'].iloc[i] > last_20['close'].iloc[i+1]:
            price_highs.append((i, last_20['close'].iloc[i]))
            macd_highs.append((i, last_20['macd_histogram'].iloc[i]))
        
        # 价格低点
        if last_20['close'].iloc[i] < last_20['close'].iloc[i-1] and last_20['close'].iloc[i] < last_20['close'].iloc[i+1]:
            price_lows.append((i, last_20['close'].iloc[i]))
            macd_lows.append((i, last_20['macd_histogram'].iloc[i]))
    
    # 分析背离
    if len(price_highs) >= 2 and len(macd_highs) >= 2:
        # 顶背离：价格创新高，MACD未创新高
        if price_highs[-1][1] > price_highs[-2][1] and macd_highs[-1][1] < macd_highs[-2][1]:
            return -1  # 看跌信号
    
    if len(price_lows) >= 2 and len(macd_lows) >= 2:
        # 底背离：价格创新低，MACD未创新低
        if price_lows[-1][1] < price_lows[-2][1] and macd_lows[-1][1] > macd_lows[-2][1]:
            return 1   # 看涨信号
    
    return 0

def make_statistical_decision(df, mean_reversion_signal, volatility_signal, 
                             momentum_signal, volume_signal, divergence_signal):
    """统计决策"""
    
    # 权重分配
    weights = {
        'mean_reversion': 0.3,  # 均值回归权重最高
        'momentum': 0.25,       # 动量次之
        'volume': 0.2,          # 成交量
        'divergence': 0.15,     # 背离
        'volatility': 0.1       # 波动率
    }
    
    # 计算加权得分
    total_score = (
        mean_reversion_signal * weights['mean_reversion'] +
        momentum_signal * weights['momentum'] +
        volume_signal * weights['volume'] +
        divergence_signal * weights['divergence']
    ) * volatility_signal  # 波动率作为调节因子
    
    # 动态阈值 - 降低阈值使预测更敏感
    threshold_high = 0.3
    threshold_medium = 0.1
    
    # 决策逻辑
    if abs(total_score) < threshold_medium:
        # 信号较弱时，根据主要信号判断
        if mean_reversion_signal > 0 or momentum_signal > 0:
            direction = "涨"
            pred_change_ratio = 0.001  # 0.1%
        elif mean_reversion_signal < 0 or momentum_signal < 0:
            direction = "跌"
            pred_change_ratio = -0.001  # -0.1%
        else:
            direction = "平"
            pred_change_ratio = 0.0
        confidence = "低"
    elif total_score >= threshold_high:
        direction = "涨"
        pred_change_ratio = 0.002  # 0.2%
        confidence = "高"
    elif total_score >= threshold_medium:
        direction = "涨"
        pred_change_ratio = 0.001  # 0.1%
        confidence = "中"
    elif total_score <= -threshold_high:
        direction = "跌"
        pred_change_ratio = -0.002  # -0.2%
        confidence = "高"
    elif total_score <= -threshold_medium:
        direction = "跌"
        pred_change_ratio = -0.001  # -0.1%
        confidence = "中"
    else:
        # 信号中等时，根据主要信号判断
        if mean_reversion_signal > 0 or momentum_signal > 0:
            direction = "涨"
            pred_change_ratio = 0.001
        elif mean_reversion_signal < 0 or momentum_signal < 0:
            direction = "跌"
            pred_change_ratio = -0.001
        else:
            direction = "平"
            pred_change_ratio = 0.0
        confidence = "中"
    
    # 计算预测价格
    last_close = df.iloc[-1]['close']
    pred_close = last_close * (1 + pred_change_ratio)
    
    # 生成分析报告
    analysis = f"🤖 统计预测分析：\n"
    analysis += f"📊 均值回归信号: {mean_reversion_signal}\n"
    analysis += f"📈 动量信号: {momentum_signal}\n"
    analysis += f"📊 成交量信号: {volume_signal}\n"
    analysis += f"📈 背离信号: {divergence_signal}\n"
    analysis += f"📊 波动率调节: {volatility_signal:.2f}\n"
    analysis += f"🎯 综合得分: {total_score:.3f}\n"
    analysis += f"🎯 预测方向: {direction} (置信度: {confidence})\n"
    
    return {
        "预测收盘价": pred_close,
        "上次收盘价": last_close,
        "预测涨跌": direction,
        "预测时间": (pd.to_datetime(df.iloc[-1]["timestamp"]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "涨跌幅度": f"{pred_change_ratio*100:.2f}%",
        "置信度": confidence,
        "综合得分": total_score,
        "均值回归信号": mean_reversion_signal,
        "动量信号": momentum_signal,
        "成交量信号": volume_signal,
        "背离信号": divergence_signal,
        "分析原因": analysis
    }

if __name__ == "__main__":
    # 加载数据
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
    
    # 打印最后一根K线
    print("【最后一根K线】")
    print(df.iloc[-1][['timestamp', 'open', 'high', 'low', 'close', 'volume']])
    
    # 预测下一根K线
    result = calculate_statistical_prediction(df)
    print("\n【统计预测结果】")
    print(result) 