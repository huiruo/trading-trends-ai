# hybrid_predictor.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from technical_indicators import add_technical_indicators
from config_improved import DATA_PATH
from advanced_rule_predictor import calculate_advanced_prediction

def calculate_hybrid_prediction(df, silent=False):
    """混合预测器 - 结合规则预测和机器学习特征"""
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 获取最新的技术指标
    latest = df.iloc[-1]
    
    # 1. 规则预测部分
    rule_result = calculate_advanced_prediction(df, silent=True)
    rule_direction = rule_result['预测涨跌']
    rule_confidence = rule_result['置信度']
    rule_signal = rule_result['综合信号']
    
    # 2. 机器学习特征分析
    ml_features = extract_ml_features(df)
    
    # 3. 市场微观结构分析
    microstructure = analyze_microstructure(df)
    
    # 4. 价格模式识别
    pattern_signal = identify_price_patterns(df)
    
    # 5. 综合决策
    final_decision = make_hybrid_decision(
        rule_result, ml_features, microstructure, pattern_signal, df
    )
    
    # 只在非静默模式下输出预测方向
    if not silent:
        pred_change_ratio_pct = final_decision['涨跌幅度'].replace('%', '')
        print(f"🔍 预测方向: {final_decision['预测涨跌']}, 变化幅度: {pred_change_ratio_pct}%")
    
    return final_decision

def extract_ml_features(df):
    """提取机器学习特征"""
    features = {}
    
    # 价格特征
    last_20_closes = df['close'].tail(20).values
    last_10_closes = df['close'].tail(10).values
    last_5_closes = df['close'].tail(5).values
    
    # 价格动量特征
    features['price_momentum_1'] = (last_5_closes[-1] - last_5_closes[-2]) / last_5_closes[-2]
    features['price_momentum_3'] = (last_5_closes[-1] - last_5_closes[-3]) / last_5_closes[-3]
    features['price_momentum_5'] = (last_5_closes[-1] - last_5_closes[0]) / last_5_closes[0]
    features['price_momentum_10'] = (last_10_closes[-1] - last_10_closes[0]) / last_10_closes[0]
    
    # 价格位置特征
    features['price_position_5'] = (last_5_closes[-1] - np.min(last_5_closes)) / (np.max(last_5_closes) - np.min(last_5_closes))
    features['price_position_10'] = (last_10_closes[-1] - np.min(last_10_closes)) / (np.max(last_10_closes) - np.min(last_10_closes))
    features['price_position_20'] = (last_20_closes[-1] - np.min(last_20_closes)) / (np.max(last_20_closes) - np.min(last_20_closes))
    
    # 波动性特征
    features['volatility_5'] = np.std(last_5_closes) / np.mean(last_5_closes)
    features['volatility_10'] = np.std(last_10_closes) / np.mean(last_10_closes)
    features['volatility_20'] = np.std(last_20_closes) / np.mean(last_20_closes)
    
    # 成交量特征
    last_10_volumes = df['volume'].tail(10).values
    features['volume_momentum'] = (last_10_volumes[-1] - last_10_volumes[0]) / last_10_volumes[0]
    features['volume_volatility'] = np.std(last_10_volumes) / np.mean(last_10_volumes)
    
    # 技术指标特征
    latest = df.iloc[-1]
    features['rsi'] = latest['rsi_14']
    features['bb_position'] = latest['bb_position']
    features['macd_histogram'] = latest['macd_histogram']
    features['ma5_ratio'] = latest['ma5_ratio']
    features['ma10_ratio'] = latest['ma10_ratio']
    
    return features

def analyze_microstructure(df):
    """分析市场微观结构"""
    microstructure = {}
    
    # 计算买卖压力
    last_5 = df.tail(5)
    
    # 价格效率（价格变化的连续性）
    price_changes = last_5['close'].pct_change().dropna()
    microstructure['price_efficiency'] = np.abs(price_changes).mean()
    
    # 成交量价格关系
    volume_price_corr = np.corrcoef(last_5['volume'], last_5['close'])[0, 1]
    microstructure['volume_price_correlation'] = volume_price_corr if not np.isnan(volume_price_corr) else 0
    
    # 价格跳跃检测
    price_jumps = np.abs(price_changes) > price_changes.std() * 2
    microstructure['price_jump_probability'] = price_jumps.mean()
    
    # 成交量异常检测
    volume_mean = last_5['volume'].mean()
    volume_std = last_5['volume'].std()
    volume_anomaly = (last_5['volume'].iloc[-1] - volume_mean) / volume_std
    microstructure['volume_anomaly'] = volume_anomaly
    
    return microstructure

def identify_price_patterns(df):
    """识别价格模式"""
    pattern_signal = 0
    
    last_10_closes = df['close'].tail(10).values
    
    # 双底模式
    if len(last_10_closes) >= 10:
        # 寻找双底
        min_indices = []
        for i in range(1, len(last_10_closes) - 1):
            if last_10_closes[i] < last_10_closes[i-1] and last_10_closes[i] < last_10_closes[i+1]:
                min_indices.append(i)
        
        if len(min_indices) >= 2:
            # 检查是否形成双底
            if abs(last_10_closes[min_indices[-1]] - last_10_closes[min_indices[-2]]) / last_10_closes[min_indices[-2]] < 0.01:
                pattern_signal += 2  # 双底看涨信号
    
    # 双顶模式
    if len(last_10_closes) >= 10:
        # 寻找双顶
        max_indices = []
        for i in range(1, len(last_10_closes) - 1):
            if last_10_closes[i] > last_10_closes[i-1] and last_10_closes[i] > last_10_closes[i+1]:
                max_indices.append(i)
        
        if len(max_indices) >= 2:
            # 检查是否形成双顶
            if abs(last_10_closes[max_indices[-1]] - last_10_closes[max_indices[-2]]) / last_10_closes[max_indices[-2]] < 0.01:
                pattern_signal -= 2  # 双顶看跌信号
    
    # 趋势线突破
    if len(last_10_closes) >= 8:
        # 简单趋势线分析
        x = np.arange(len(last_10_closes))
        slope, intercept = np.polyfit(x, last_10_closes, 1)
        trend_line = slope * x + intercept
        
        # 检查是否突破趋势线
        current_price = last_10_closes[-1]
        expected_price = trend_line[-1]
        
        if current_price > expected_price * 1.005:  # 向上突破
            pattern_signal += 1
        elif current_price < expected_price * 0.995:  # 向下突破
            pattern_signal -= 1
    
    return pattern_signal

def make_hybrid_decision(rule_result, ml_features, microstructure, pattern_signal, df):
    """综合决策"""
    
    # 规则预测权重
    rule_weight = 0.4
    ml_weight = 0.3
    microstructure_weight = 0.2
    pattern_weight = 0.1
    
    # 机器学习特征得分
    ml_score = calculate_ml_score(ml_features)
    
    # 微观结构得分
    microstructure_score = calculate_microstructure_score(microstructure)
    
    # 综合得分
    total_score = (
        rule_result['综合信号'] * rule_weight +
        ml_score * ml_weight +
        microstructure_score * microstructure_weight +
        pattern_signal * pattern_weight
    )
    
    # 动态阈值（根据市场条件调整）
    volatility = ml_features['volatility_10']
    if volatility > 0.03:  # 高波动性
        threshold_high = 1.0
        threshold_medium = 0.5
    else:  # 低波动性
        threshold_high = 1.3
        threshold_medium = 0.7
    
    # 决策逻辑
    if abs(total_score) < threshold_medium:
        # 信号较弱，倾向于延续当前趋势
        if rule_result['趋势一致性'] > 0:
            direction = "涨"
            pred_change_ratio = 0.001
        elif rule_result['趋势一致性'] < 0:
            direction = "跌"
            pred_change_ratio = -0.001
        else:
            direction = "平"
            pred_change_ratio = 0.0
        confidence = "低"
    elif total_score >= threshold_high:
        direction = "涨"
        pred_change_ratio = 0.003
        confidence = "高"
    elif total_score >= threshold_medium:
        direction = "涨"
        pred_change_ratio = 0.002
        confidence = "中"
    elif total_score <= -threshold_high:
        direction = "跌"
        pred_change_ratio = -0.003
        confidence = "高"
    elif total_score <= -threshold_medium:
        direction = "跌"
        pred_change_ratio = -0.002
        confidence = "中"
    else:
        # 信号中等，根据规则预测判断
        if rule_result['趋势一致性'] > 0:
            direction = "涨"
            pred_change_ratio = 0.001
        elif rule_result['趋势一致性'] < 0:
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
    analysis = f"🤖 混合预测分析：\n"
    analysis += f"📊 规则预测: {rule_result['预测涨跌']} (信号:{rule_result['综合信号']:.3f})\n"
    analysis += f"📈 机器学习特征得分: {ml_score:.3f}\n"
    analysis += f"📊 微观结构得分: {microstructure_score:.3f}\n"
    analysis += f"📈 价格模式信号: {pattern_signal}\n"
    analysis += f"🎯 综合得分: {total_score:.3f}\n"
    analysis += f"🎯 最终预测: {direction} (置信度: {confidence})\n"
    
    return {
        "预测收盘价": pred_close,
        "上次收盘价": last_close,
        "预测涨跌": direction,
        "预测时间": (pd.to_datetime(df.iloc[-1]["timestamp"]) + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        "涨跌幅度": f"{pred_change_ratio*100:.2f}%",
        "置信度": confidence,
        "综合得分": total_score,
        "规则信号": rule_result['综合信号'],
        "机器学习得分": ml_score,
        "微观结构得分": microstructure_score,
        "模式信号": pattern_signal,
        "分析原因": analysis
    }

def calculate_ml_score(features):
    """计算机器学习特征得分"""
    score = 0
    
    # 价格动量得分
    if features['price_momentum_1'] > 0.001:
        score += 1
    elif features['price_momentum_1'] < -0.001:
        score -= 1
    
    if features['price_momentum_3'] > 0.002:
        score += 1
    elif features['price_momentum_3'] < -0.002:
        score -= 1
    
    # 价格位置得分
    if features['price_position_5'] < 0.3:
        score += 1  # 接近底部
    elif features['price_position_5'] > 0.7:
        score -= 1  # 接近顶部
    
    # RSI得分
    if features['rsi'] < 30:
        score += 2  # 超卖
    elif features['rsi'] > 70:
        score -= 2  # 超买
    elif features['rsi'] < 40:
        score += 1
    elif features['rsi'] > 60:
        score -= 1
    
    # 布林带得分
    if features['bb_position'] < 0.2:
        score += 1  # 接近下轨
    elif features['bb_position'] > 0.8:
        score -= 1  # 接近上轨
    
    # MACD得分
    if features['macd_histogram'] > 0:
        score += 1
    else:
        score -= 1
    
    return score / 10  # 归一化

def calculate_microstructure_score(microstructure):
    """计算微观结构得分"""
    score = 0
    
    # 价格效率得分
    if microstructure['price_efficiency'] < 0.01:
        score += 1  # 价格变化平稳
    elif microstructure['price_efficiency'] > 0.03:
        score -= 1  # 价格变化剧烈
    
    # 成交量价格相关性得分
    if microstructure['volume_price_correlation'] > 0.5:
        score += 1  # 价量配合
    elif microstructure['volume_price_correlation'] < -0.5:
        score -= 1  # 价量背离
    
    # 成交量异常得分
    if microstructure['volume_anomaly'] > 1:
        score += 1  # 放量
    elif microstructure['volume_anomaly'] < -1:
        score -= 1  # 缩量
    
    return score / 3  # 归一化

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
    result = calculate_hybrid_prediction(df)
    print("\n【混合预测结果】")
    print(result) 