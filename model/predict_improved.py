# model/predict_improved.py - 改进版预测脚本
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from model.model import LSTMModel
from preprocess import load_and_preprocess, create_sequences, load_scaler, inverse_transform_close
from technical_indicators import add_technical_indicators
from config_improved import *

def predict_next_candle_improved(df: pd.DataFrame):
    # 加载已保存的scaler
    loaded_scaler = load_scaler()
    if loaded_scaler is None:
        print("⚠️ No scaler found. 请先重新训练模型 (python -m model.train_improved)")
        return None
    
    # 检查scaler特征名与当前特征名是否一致
    scaler_features = getattr(loaded_scaler, 'feature_names_in_', None)
    if scaler_features is not None:
        from config_improved import FEATURE_COLUMNS
        if list(scaler_features) != list(FEATURE_COLUMNS):
            print("❌ 检测到 scaler.pkl 特征与当前 FEATURE_COLUMNS 不一致！")
            print(f"scaler特征: {list(scaler_features)}")
            print(f"当前特征: {list(FEATURE_COLUMNS)}")
            print("请删除 model/scaler.pkl 并重新训练模型: python -m model.train_improved")
            return None
    
    # 添加技术指标
    df_with_indicators = add_technical_indicators(df)
    
    # 使用已保存的scaler进行标准化（对所有特征）
    df_processed = df_with_indicators.copy()
    df_processed[FEATURE_COLUMNS] = loaded_scaler.transform(df_with_indicators[FEATURE_COLUMNS])

    X, y = create_sequences(df_processed, window_size=WINDOW_SIZE)
    
    if len(X) == 0:
        print("⚠️ Not enough data for prediction. Need at least {} data points.".format(WINDOW_SIZE + 1))
        return None

    X_latest = X[-1]
    X_tensor = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0)

    model = LSTMModel(input_size=X_tensor.shape[2], hidden_size=32, num_layers=2, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        pred_probs = model(X_tensor)  # 得到分类概率
        pred_class = torch.argmax(pred_probs, dim=1).item()  # 得到预测类别

    # 根据分类结果计算预测价格
    last_close = df.iloc[-1]['close']
    
    # 分类结果：0=跌，1=平，2=涨
    if pred_class == 0:  # 跌
        pred_close = last_close * 0.999  # 微跌0.1%
        direction = "跌"
    elif pred_class == 2:  # 涨
        pred_close = last_close * 1.001  # 微涨0.1%
        direction = "涨"
    else:  # 平
        pred_close = last_close  # 不变
        direction = "平"
    
    pred_change_ratio = (pred_close - last_close) / last_close
    print(f"🔍 预测方向: {direction}, 变化幅度: {pred_change_ratio*100:.3f}%")

    last_close_time = pd.to_datetime(df.iloc[-1]["timestamp"])
    pred_time = last_close_time + pd.Timedelta(hours=1)

    # 分析预测原因
    analysis = analyze_prediction_reason(df_with_indicators, pred_change_ratio, direction)

    return {
        "预测收盘价": pred_close,
        "上次收盘价": last_close,
        "预测涨跌": direction,
        "预测时间": pred_time.strftime("%Y-%m-%d %H:%M:%S"),
        "涨跌幅度": f"{((pred_close - last_close) / last_close * 100):.2f}%",
        "分析原因": analysis
    }

def analyze_prediction_reason(df: pd.DataFrame, change_ratio: float, direction: str):
    """分析预测结果的原因"""
    
    # 获取最后几根K线的数据
    last_5 = df.tail(5)
    
    # 分析技术指标
    last_row = df.iloc[-1]
    
    analysis = f"🤖 AI分析预测原因：\n"
    
    # 1. 价格趋势分析
    price_trend = analyze_price_trend(last_5)
    analysis += f"📈 价格趋势：{price_trend}\n"
    
    # 2. RSI分析
    rsi_analysis = analyze_rsi(last_row['rsi_14'])
    analysis += f"📊 RSI指标：{rsi_analysis}\n"
    
    # 3. 布林带分析
    bb_analysis = analyze_bollinger_bands(last_row['bb_position'])
    analysis += f"📉 布林带位置：{bb_analysis}\n"
    
    # 4. MACD分析
    if 'macd_histogram' in last_row:
        macd_analysis = analyze_macd(last_row['macd_histogram'])
        analysis += f"📊 MACD指标：{macd_analysis}\n"
    
    # 5. KDJ分析
    if 'kdj_k' in last_row and 'kdj_d' in last_row:
        kdj_analysis = analyze_kdj(last_row['kdj_k'], last_row['kdj_d'], last_row['kdj_j'])
        analysis += f"📊 KDJ指标：{kdj_analysis}\n"
    
    # 6. 移动平均线分析
    if 'ma5_ratio' in last_row and 'ma10_ratio' in last_row:
        ma_analysis = analyze_moving_averages(last_row['ma5_ratio'], last_row['ma10_ratio'], last_row['ma20_ratio'])
        analysis += f"📈 移动平均线：{ma_analysis}\n"
    
    # 7. 成交量分析
    volume_analysis = analyze_volume(last_5)
    analysis += f"📊 成交量趋势：{volume_analysis}\n"
    
    # 8. 动量分析
    if 'momentum_5' in last_row:
        momentum_analysis = analyze_momentum(last_row['momentum_5'], last_row['momentum_10'])
        analysis += f"📊 动量指标：{momentum_analysis}\n"
    
    # 9. 综合判断
    overall_analysis = get_overall_analysis(change_ratio, direction, last_row)
    analysis += f"🎯 综合判断：{overall_analysis}\n"
    
    return analysis

def analyze_price_trend(last_5: pd.DataFrame):
    """分析价格趋势"""
    closes = last_5['close'].values
    if len(closes) < 3:
        return "数据不足"
    
    # 计算最近3根K线的趋势
    recent_trend = (closes[-1] - closes[-3]) / closes[-3] * 100
    
    if recent_trend > 0.5:
        return f"强势上涨趋势 (+{recent_trend:.2f}%)"
    elif recent_trend > 0.1:
        return f"温和上涨趋势 (+{recent_trend:.2f}%)"
    elif recent_trend < -0.5:
        return f"强势下跌趋势 ({recent_trend:.2f}%)"
    elif recent_trend < -0.1:
        return f"温和下跌趋势 ({recent_trend:.2f}%)"
    else:
        return f"横盘整理 ({recent_trend:.2f}%)"

def analyze_rsi(rsi: float):
    """分析RSI指标"""
    if rsi > 70:
        return f"超买区域 ({rsi:.1f})，可能回调"
    elif rsi > 60:
        return f"偏强区域 ({rsi:.1f})，上涨动能较强"
    elif rsi < 30:
        return f"超卖区域 ({rsi:.1f})，可能反弹"
    elif rsi < 40:
        return f"偏弱区域 ({rsi:.1f})，下跌压力较大"
    else:
        return f"中性区域 ({rsi:.1f})，无明显方向"

def analyze_bollinger_bands(bb_position: float):
    """分析布林带位置"""
    if bb_position > 0.8:
        return f"接近上轨 ({bb_position:.3f})，可能遇阻回落"
    elif bb_position > 0.6:
        return f"偏上位置 ({bb_position:.3f})，上涨空间有限"
    elif bb_position < 0.2:
        return f"接近下轨 ({bb_position:.3f})，可能获得支撑"
    elif bb_position < 0.4:
        return f"偏下位置 ({bb_position:.3f})，下跌空间有限"
    else:
        return f"中轨附近 ({bb_position:.3f})，方向不明"

def analyze_volume(last_5: pd.DataFrame):
    """分析成交量趋势"""
    volumes = last_5['volume'].values
    if len(volumes) < 3:
        return "数据不足"
    
    recent_avg = volumes[-3:].mean()
    current_volume = volumes[-1]
    
    volume_ratio = current_volume / recent_avg
    
    if volume_ratio > 1.5:
        return f"成交量放大 ({volume_ratio:.2f}倍)，市场活跃"
    elif volume_ratio > 1.2:
        return f"成交量增加 ({volume_ratio:.2f}倍)，交投活跃"
    elif volume_ratio < 0.7:
        return f"成交量萎缩 ({volume_ratio:.2f}倍)，市场观望"
    else:
        return f"成交量正常 ({volume_ratio:.2f}倍)，交投平稳"

def get_overall_analysis(change_ratio: float, direction: str, last_row: pd.Series):
    """综合判断"""
    abs_change = abs(change_ratio) * 100
    
    if abs_change > 3:
        intensity = "较大"
        confidence = "中等"
    elif abs_change > 1.5:
        intensity = "中等"
        confidence = "较高"
    elif abs_change > 0.5:
        intensity = "温和"
        confidence = "高"
    else:
        intensity = "轻微"
        confidence = "很高"
    
    if direction == "涨":
        return f"基于技术指标综合分析，预计将出现{intensity}上涨，置信度{confidence}，建议关注支撑位"
    else:
        return f"基于技术指标综合分析，预计将出现{intensity}下跌，置信度{confidence}，建议关注阻力位"

def analyze_macd(macd_histogram: float):
    """分析MACD指标"""
    if macd_histogram > 0:
        if macd_histogram > 100:
            return f"强势上涨信号 ({macd_histogram:.2f})"
        else:
            return f"温和上涨信号 ({macd_histogram:.2f})"
    else:
        if macd_histogram < -100:
            return f"强势下跌信号 ({macd_histogram:.2f})"
        else:
            return f"温和下跌信号 ({macd_histogram:.2f})"

def analyze_kdj(k: float, d: float, j: float):
    """分析KDJ指标"""
    if k > 80 and d > 80:
        return f"超买区域 (K:{k:.1f}, D:{d:.1f}, J:{j:.1f})，可能回调"
    elif k < 20 and d < 20:
        return f"超卖区域 (K:{k:.1f}, D:{d:.1f}, J:{j:.1f})，可能反弹"
    elif k > d:
        return f"金叉信号 (K:{k:.1f}, D:{d:.1f}, J:{j:.1f})，上涨概率较大"
    else:
        return f"死叉信号 (K:{k:.1f}, D:{d:.1f}, J:{j:.1f})，下跌概率较大"

def analyze_moving_averages(ma5_ratio: float, ma10_ratio: float, ma20_ratio: float):
    """分析移动平均线"""
    if ma5_ratio > 1.01 and ma10_ratio > 1.01:
        return f"强势上涨 (MA5:{ma5_ratio:.3f}, MA10:{ma10_ratio:.3f}, MA20:{ma20_ratio:.3f})"
    elif ma5_ratio < 0.99 and ma10_ratio < 0.99:
        return f"强势下跌 (MA5:{ma5_ratio:.3f}, MA10:{ma10_ratio:.3f}, MA20:{ma20_ratio:.3f})"
    elif ma5_ratio > ma10_ratio > ma20_ratio:
        return f"多头排列 (MA5:{ma5_ratio:.3f}, MA10:{ma10_ratio:.3f}, MA20:{ma20_ratio:.3f})"
    elif ma5_ratio < ma10_ratio < ma20_ratio:
        return f"空头排列 (MA5:{ma5_ratio:.3f}, MA10:{ma10_ratio:.3f}, MA20:{ma20_ratio:.3f})"
    else:
        return f"震荡整理 (MA5:{ma5_ratio:.3f}, MA10:{ma10_ratio:.3f}, MA20:{ma20_ratio:.3f})"

def analyze_momentum(momentum_5: float, momentum_10: float):
    """分析动量指标"""
    if momentum_5 > 0.02 and momentum_10 > 0.02:
        return f"强势上涨动量 (5日:{momentum_5*100:.2f}%, 10日:{momentum_10*100:.2f}%)"
    elif momentum_5 < -0.02 and momentum_10 < -0.02:
        return f"强势下跌动量 (5日:{momentum_5*100:.2f}%, 10日:{momentum_10*100:.2f}%)"
    elif momentum_5 > 0:
        return f"短期上涨动量 (5日:{momentum_5*100:.2f}%, 10日:{momentum_10*100:.2f}%)"
    else:
        return f"短期下跌动量 (5日:{momentum_5*100:.2f}%, 10日:{momentum_10*100:.2f}%)"

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    # 重命名列以匹配预处理函数
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
    result = predict_next_candle_improved(df)
    if result:
        print("\n【预测的下一根K线】")
        print(result)
        # 如果数据集里有真实的下一根K线，也打印出来
        next_time = pd.to_datetime(result["预测时间"])
        real_next = df[df['timestamp'] == next_time]
        if not real_next.empty:
            print("\n【真实的下一根K线】")
            print(real_next.iloc[0][['timestamp', 'open', 'high', 'low', 'close', 'volume']])
        else:
            print("\n【真实的下一根K线】")
            print("数据集中没有下一根K线（可能是最新一根）")
    else:
        print("预测失败，请先训练模型") 