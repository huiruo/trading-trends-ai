# simple_rule_predictor.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from technical_indicators import add_technical_indicators
from config_improved import DATA_PATH

# python simple_rule_predictor/simple_rule_predictor.py
# 规则预测器比深度学习模型更有效，原因：
# 更稳定 - 不依赖复杂的神经网络训练
# 更可解释 - 每个预测都有明确的技术指标依据，更容易理解
# 更实用 - 预测幅度更现实，适合实际交易
# 更快速 - 计算速度快，适合实时交易
# 更可靠 - 基于历史数据分析，更可靠
def calculate_simple_prediction(df, silent=False):
    """基于技术指标的简单规则预测"""
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 获取最新的技术指标
    latest = df.iloc[-1]
    
    # 计算各种信号
    signals = []
    
    # 1. RSI信号
    rsi = latest['rsi_14']
    if rsi < 30:
        signals.append(('RSI超卖', 1))  # 看涨信号
    elif rsi > 70:
        signals.append(('RSI超买', -1))  # 看跌信号
    else:
        signals.append(('RSI中性', 0))
    
    # 2. 布林带信号
    bb_pos = latest['bb_position']
    if bb_pos < 0.2:
        signals.append(('布林带下轨', 1))  # 看涨信号
    elif bb_pos > 0.8:
        signals.append(('布林带上轨', -1))  # 看跌信号
    else:
        signals.append(('布林带中轨', 0))
    
    # 3. MACD信号
    macd = latest['macd_histogram']
    if macd > 0:
        signals.append(('MACD正', 1))  # 看涨信号
    else:
        signals.append(('MACD负', -1))  # 看跌信号
    
    # 4. 移动平均线信号
    ma5_ratio = latest['ma5_ratio']
    if ma5_ratio > 1.001:
        signals.append(('MA5上方', 1))  # 看涨信号
    elif ma5_ratio < 0.999:
        signals.append(('MA5下方', -1))  # 看跌信号
    else:
        signals.append(('MA5附近', 0))
    
    # 5. 成交量信号
    volume_ratio = latest['volume_ratio_5']
    if volume_ratio > 1.2:
        signals.append(('成交量放大', 1))  # 看涨信号
    elif volume_ratio < 0.8:
        signals.append(('成交量萎缩', -1))  # 看跌信号
    else:
        signals.append(('成交量正常', 0))
    
    # 6. 价格趋势信号
    last_5_closes = df['close'].tail(5).values
    if len(last_5_closes) >= 3:
        recent_trend = (last_5_closes[-1] - last_5_closes[-3]) / last_5_closes[-3]
        if recent_trend > 0.005:  # 0.5%以上
            signals.append(('价格上升趋势', 1))
        elif recent_trend < -0.005:  # -0.5%以下
            signals.append(('价格下降趋势', -1))
        else:
            signals.append(('价格横盘', 0))
    
    # 计算综合信号
    total_signal = sum(signal[1] for signal in signals)
    
    # 根据信号强度判断方向
    if total_signal >= 2:
        direction = "涨"
        confidence = "高"
    elif total_signal <= -2:
        direction = "跌"
        confidence = "高"
    elif total_signal == 1:
        direction = "涨"
        confidence = "中"
    elif total_signal == -1:
        direction = "跌"
        confidence = "中"
    else:
        direction = "平"
        confidence = "低"
    
    # 计算预测价格变化
    last_close = df.iloc[-1]['close']
    if direction == "涨":
        pred_change_ratio = 0.001  # 0.1%
    elif direction == "跌":
        pred_change_ratio = -0.001  # -0.1%
    else:
        pred_change_ratio = 0.0
    
    pred_close = last_close * (1 + pred_change_ratio)
    
    # 只在非静默模式下输出预测方向
    if not silent:
        pred_change_ratio_pct = pred_change_ratio * 100
        print(f"🔍 预测方向: {direction}, 变化幅度: {pred_change_ratio_pct:.3f}%")
    
    # 生成分析报告
    analysis = f"🤖 规则预测分析：\n"
    analysis += f"📊 综合信号强度: {total_signal}\n"
    analysis += f"🎯 预测方向: {direction} (置信度: {confidence})\n\n"
    
    analysis += "📈 各指标信号：\n"
    for signal_name, signal_value in signals:
        if signal_value > 0:
            analysis += f"  ✅ {signal_name}: 看涨\n"
        elif signal_value < 0:
            analysis += f"  ❌ {signal_name}: 看跌\n"
        else:
            analysis += f"  ➖ {signal_name}: 中性\n"
    
    return {
        "预测收盘价": pred_close,
        "上次收盘价": last_close,
        "预测涨跌": direction,
        "预测时间": (pd.to_datetime(df.iloc[-1]["timestamp"]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "涨跌幅度": f"{pred_change_ratio*100:.2f}%",
        "置信度": confidence,
        "综合信号": total_signal,
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
    result = calculate_simple_prediction(df)
    print("\n【规则预测结果】")
    print(result) 