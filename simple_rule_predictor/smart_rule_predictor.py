# smart_rule_predictor.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from technical_indicators import add_technical_indicators
from config_improved import DATA_PATH

def calculate_smart_prediction(df, silent=False):
    """智能规则预测器 - 基于市场趋势连续性和动态阈值"""
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 获取最新的技术指标
    latest = df.iloc[-1]
    
    # 分析最近的价格趋势
    last_10_closes = df['close'].tail(10).values
    last_5_closes = df['close'].tail(5).values
    
    # 计算多个时间框架的趋势
    if len(last_10_closes) >= 5:
        # 短期趋势（3根K线）
        short_trend = (last_5_closes[-1] - last_5_closes[-3]) / last_5_closes[-3]
        # 中期趋势（5根K线）
        medium_trend = (last_10_closes[-1] - last_10_closes[-5]) / last_10_closes[-5]
        # 长期趋势（10根K线）
        long_trend = (last_10_closes[-1] - last_10_closes[-10]) / last_10_closes[-10]
    else:
        short_trend = medium_trend = long_trend = 0
    
    # 计算趋势一致性
    trend_consistency = 0
    if short_trend > 0 and medium_trend > 0 and long_trend > 0:
        trend_consistency = 3  # 强势上涨
    elif short_trend > 0 and medium_trend > 0:
        trend_consistency = 2  # 温和上涨
    elif short_trend > 0:
        trend_consistency = 1  # 短期上涨
    elif short_trend < 0 and medium_trend < 0 and long_trend < 0:
        trend_consistency = -3  # 强势下跌
    elif short_trend < 0 and medium_trend < 0:
        trend_consistency = -2  # 温和下跌
    elif short_trend < 0:
        trend_consistency = -1  # 短期下跌
    else:
        trend_consistency = 0  # 震荡
    
    # 分析技术指标
    signals = []
    
    # 1. RSI分析（考虑趋势）
    rsi = latest['rsi_14']
    rsi_prev = df.iloc[-2]['rsi_14'] if len(df) > 1 else rsi
    rsi_trend = rsi - rsi_prev
    
    if trend_consistency > 0:  # 上涨趋势中
        if rsi < 40:  # 上涨趋势中RSI偏低，看涨
            rsi_signal = 2
        elif rsi < 60:
            rsi_signal = 1
        else:
            rsi_signal = 0
    elif trend_consistency < 0:  # 下跌趋势中
        if rsi > 60:  # 下跌趋势中RSI偏高，看跌
            rsi_signal = -2
        elif rsi > 40:
            rsi_signal = -1
        else:
            rsi_signal = 0
    else:  # 震荡趋势
        if rsi < 30:
            rsi_signal = 1
        elif rsi > 70:
            rsi_signal = -1
        else:
            rsi_signal = 0
    
    signals.append(('RSI', rsi_signal, 1.0))
    
    # 2. 布林带分析
    bb_pos = latest['bb_position']
    bb_pos_prev = df.iloc[-2]['bb_position'] if len(df) > 1 else bb_pos
    bb_trend = bb_pos - bb_pos_prev
    
    if bb_pos < 0.2 and bb_trend > 0:
        bb_signal = 2  # 下轨反弹
    elif bb_pos < 0.3:
        bb_signal = 1  # 接近下轨
    elif bb_pos > 0.8 and bb_trend < 0:
        bb_signal = -2  # 上轨回落
    elif bb_pos > 0.7:
        bb_signal = -1  # 接近上轨
    else:
        bb_signal = 0
    
    signals.append(('布林带', bb_signal, 1.1))
    
    # 3. MACD分析
    macd = latest['macd_histogram']
    macd_prev = df.iloc[-2]['macd_histogram'] if len(df) > 1 else macd
    macd_trend = macd - macd_prev
    
    if macd > 0 and macd_trend > 0:
        macd_signal = 2  # 强势上涨
    elif macd > 0:
        macd_signal = 1  # 温和上涨
    elif macd < 0 and macd_trend < 0:
        macd_signal = -2  # 强势下跌
    elif macd < 0:
        macd_signal = -1  # 温和下跌
    else:
        macd_signal = 0
    
    signals.append(('MACD', macd_signal, 1.2))
    
    # 4. 成交量分析
    volume_ratio = latest['volume_ratio_5']
    if volume_ratio > 1.3:
        volume_signal = 1  # 放量
    elif volume_ratio < 0.7:
        volume_signal = -1  # 缩量
    else:
        volume_signal = 0
    
    signals.append(('成交量', volume_signal, 0.8))
    
    # 5. 移动平均线分析
    ma5_ratio = latest['ma5_ratio']
    ma10_ratio = latest['ma10_ratio']
    
    if ma5_ratio > 1.002 and ma10_ratio > 1.001:
        ma_signal = 1  # 多头排列
    elif ma5_ratio < 0.998 and ma10_ratio < 0.999:
        ma_signal = -1  # 空头排列
    else:
        ma_signal = 0
    
    signals.append(('移动平均线', ma_signal, 0.9))
    
    # 计算综合信号
    total_signal = sum(signal[1] * signal[2] for signal in signals)
    total_weight = sum(signal[2] for signal in signals)
    normalized_signal = total_signal / total_weight
    
    # 结合趋势一致性做最终判断
    final_signal = normalized_signal + trend_consistency * 0.3
    
    # 动态阈值判断
    if abs(final_signal) < 0.5:
        # 信号较弱，倾向于延续当前趋势
        if trend_consistency > 0:
            direction = "涨"
            pred_change_ratio = 0.001  # 0.1%
        elif trend_consistency < 0:
            direction = "跌"
            pred_change_ratio = -0.001  # -0.1%
        else:
            direction = "平"
            pred_change_ratio = 0.0
        confidence = "低"
    elif final_signal >= 1.5:
        direction = "涨"
        pred_change_ratio = 0.003  # 0.3%
        confidence = "高"
    elif final_signal >= 0.8:
        direction = "涨"
        pred_change_ratio = 0.002  # 0.2%
        confidence = "中"
    elif final_signal <= -1.5:
        direction = "跌"
        pred_change_ratio = -0.003  # -0.3%
        confidence = "高"
    elif final_signal <= -0.8:
        direction = "跌"
        pred_change_ratio = -0.002  # -0.2%
        confidence = "中"
    else:
        # 信号中等，根据趋势判断
        if trend_consistency > 0:
            direction = "涨"
            pred_change_ratio = 0.001
        elif trend_consistency < 0:
            direction = "跌"
            pred_change_ratio = -0.001
        else:
            direction = "平"
            pred_change_ratio = 0.0
        confidence = "中"
    
    # 计算预测价格
    last_close = df.iloc[-1]['close']
    pred_close = last_close * (1 + pred_change_ratio)
    
    # 只在非静默模式下输出预测方向
    if not silent:
        pred_change_ratio_pct = pred_change_ratio * 100
        print(f"🔍 预测方向: {direction}, 变化幅度: {pred_change_ratio_pct:.3f}%")
    
    # 生成分析报告
    analysis = f"🤖 智能规则预测分析：\n"
    analysis += f"📊 趋势一致性: {trend_consistency} (短期:{short_trend*100:.2f}%, 中期:{medium_trend*100:.2f}%, 长期:{long_trend*100:.2f}%)\n"
    analysis += f"📈 综合信号: {final_signal:.3f}\n"
    analysis += f"🎯 预测方向: {direction} (置信度: {confidence})\n\n"
    
    analysis += "📈 各指标信号：\n"
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
        "综合信号": final_signal,
        "趋势一致性": trend_consistency,
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
    result = calculate_smart_prediction(df)
    print("\n【智能规则预测结果】")
    print(result) 