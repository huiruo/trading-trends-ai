# simple_rule_predictor_improved.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from technical_indicators import add_technical_indicators
from config_improved import DATA_PATH

def calculate_improved_prediction(df, silent=False):
    """改进版规则预测器 - 基于动态权重和趋势分析"""
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 获取最新的技术指标
    latest = df.iloc[-1]
    
    # 计算动态信号强度
    signals = []
    
    # 1. RSI动态信号（考虑趋势）
    rsi = latest['rsi_14']
    rsi_prev = df.iloc[-2]['rsi_14'] if len(df) > 1 else rsi
    rsi_trend = rsi - rsi_prev
    
    if rsi < 25:
        rsi_signal = 2  # 强烈超卖
    elif rsi < 35 and rsi_trend > 0:
        rsi_signal = 1  # 超卖反弹
    elif rsi > 75:
        rsi_signal = -2  # 强烈超买
    elif rsi > 65 and rsi_trend < 0:
        rsi_signal = -1  # 超买回落
    else:
        rsi_signal = 0
    
    signals.append(('RSI', rsi_signal, 1.2))  # RSI权重较高
    
    # 2. 布林带动态信号
    bb_pos = latest['bb_position']
    bb_pos_prev = df.iloc[-2]['bb_position'] if len(df) > 1 else bb_pos
    bb_trend = bb_pos - bb_pos_prev
    
    if bb_pos < 0.1:
        bb_signal = 2  # 强烈支撑
    elif bb_pos < 0.3 and bb_trend > 0:
        bb_signal = 1  # 支撑反弹
    elif bb_pos > 0.9:
        bb_signal = -2  # 强烈阻力
    elif bb_pos > 0.7 and bb_trend < 0:
        bb_signal = -1  # 阻力回落
    else:
        bb_signal = 0
    
    signals.append(('布林带', bb_signal, 1.0))
    
    # 3. MACD动态信号
    macd = latest['macd_histogram']
    macd_prev = df.iloc[-2]['macd_histogram'] if len(df) > 1 else macd
    macd_trend = macd - macd_prev
    
    if macd > 50 and macd_trend > 0:
        macd_signal = 2  # 强势上涨
    elif macd > 0:
        macd_signal = 1  # 温和上涨
    elif macd < -50 and macd_trend < 0:
        macd_signal = -2  # 强势下跌
    elif macd < 0:
        macd_signal = -1  # 温和下跌
    else:
        macd_signal = 0
    
    signals.append(('MACD', macd_signal, 1.1))
    
    # 4. 价格趋势信号（多时间框架）
    last_10_closes = df['close'].tail(10).values
    if len(last_10_closes) >= 5:
        # 短期趋势（3根K线）
        short_trend = (last_10_closes[-1] - last_10_closes[-3]) / last_10_closes[-3]
        # 中期趋势（5根K线）
        medium_trend = (last_10_closes[-1] - last_10_closes[-5]) / last_10_closes[-5]
        
        if short_trend > 0.01 and medium_trend > 0.005:
            trend_signal = 2  # 强势上涨
        elif short_trend > 0.005:
            trend_signal = 1  # 温和上涨
        elif short_trend < -0.01 and medium_trend < -0.005:
            trend_signal = -2  # 强势下跌
        elif short_trend < -0.005:
            trend_signal = -1  # 温和下跌
        else:
            trend_signal = 0
    else:
        trend_signal = 0
    
    signals.append(('价格趋势', trend_signal, 1.3))  # 价格趋势权重最高
    
    # 5. 成交量确认信号
    volume_ratio = latest['volume_ratio_5']
    if volume_ratio > 1.5:
        volume_signal = 1  # 放量确认
    elif volume_ratio < 0.5:
        volume_signal = -1  # 缩量
    else:
        volume_signal = 0
    
    signals.append(('成交量', volume_signal, 0.8))
    
    # 6. 移动平均线信号
    ma5_ratio = latest['ma5_ratio']
    ma10_ratio = latest['ma10_ratio']
    
    if ma5_ratio > 1.005 and ma10_ratio > 1.002:
        ma_signal = 1  # 多头排列
    elif ma5_ratio < 0.995 and ma10_ratio < 0.998:
        ma_signal = -1  # 空头排列
    else:
        ma_signal = 0
    
    signals.append(('移动平均线', ma_signal, 0.9))
    
    # 计算加权综合信号
    weighted_signal = sum(signal[1] * signal[2] for signal in signals)
    total_weight = sum(signal[2] for signal in signals)
    normalized_signal = weighted_signal / total_weight
    
    # 根据信号强度判断方向和幅度
    if normalized_signal >= 1.0:
        direction = "涨"
        confidence = "高"
        pred_change_ratio = 0.005  # 0.5%
    elif normalized_signal >= 0.3:
        direction = "涨"
        confidence = "中"
        pred_change_ratio = 0.002  # 0.2%
    elif normalized_signal <= -1.0:
        direction = "跌"
        confidence = "高"
        pred_change_ratio = -0.005  # -0.5%
    elif normalized_signal <= -0.3:
        direction = "跌"
        confidence = "中"
        pred_change_ratio = -0.002  # -0.2%
    else:
        direction = "平"
        confidence = "低"
        pred_change_ratio = 0.0
    
    # 计算预测价格
    last_close = df.iloc[-1]['close']
    pred_close = last_close * (1 + pred_change_ratio)
    
    # 只在非静默模式下输出预测方向
    if not silent:
        pred_change_ratio_pct = pred_change_ratio * 100
        print(f"🔍 预测方向: {direction}, 变化幅度: {pred_change_ratio_pct:.3f}%")
    
    # 生成分析报告
    analysis = f"🤖 改进规则预测分析：\n"
    analysis += f"📊 加权综合信号: {normalized_signal:.3f}\n"
    analysis += f"🎯 预测方向: {direction} (置信度: {confidence})\n\n"
    
    analysis += "📈 各指标信号（信号值×权重）：\n"
    for signal_name, signal_value, weight in signals:
        weighted_value = signal_value * weight
        if signal_value > 0:
            analysis += f"  ✅ {signal_name}: +{signal_value} (权重{weight}) = {weighted_value:.2f}\n"
        elif signal_value < 0:
            analysis += f"  ❌ {signal_name}: {signal_value} (权重{weight}) = {weighted_value:.2f}\n"
        else:
            analysis += f"  ➖ {signal_name}: {signal_value} (权重{weight}) = {weighted_value:.2f}\n"
    
    return {
        "预测收盘价": pred_close,
        "上次收盘价": last_close,
        "预测涨跌": direction,
        "预测时间": (pd.to_datetime(df.iloc[-1]["timestamp"]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "涨跌幅度": f"{pred_change_ratio*100:.2f}%",
        "置信度": confidence,
        "综合信号": normalized_signal,
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
    result = calculate_improved_prediction(df)
    print("\n【改进规则预测结果】")
    print(result) 