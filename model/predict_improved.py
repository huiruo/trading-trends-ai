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
    
    # 使用已保存的scaler进行标准化
    df_processed = df_with_indicators.copy()
    df_processed[FEATURE_COLUMNS] = loaded_scaler.transform(df_with_indicators[FEATURE_COLUMNS])

    X, y = create_sequences(df_processed, window_size=WINDOW_SIZE)
    
    if len(X) == 0:
        print("⚠️ Not enough data for prediction. Need at least {} data points.".format(WINDOW_SIZE + 1))
        return None

    X_latest = X[-1]
    X_tensor = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0)

    model = LSTMModel(input_size=X_tensor.shape[2], hidden_size=16, num_layers=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        pred_normalized = model(X_tensor).item()

    # 检查是否使用相对变化目标
    try:
        from config_improved import USE_RELATIVE_CHANGE, MAX_CHANGE_RATIO
    except ImportError:
        USE_RELATIVE_CHANGE = False
        MAX_CHANGE_RATIO = 0.05
    
    if USE_RELATIVE_CHANGE:
        # 从归一化的相对变化转换为实际变化幅度
        pred_change_ratio = (pred_normalized * 2 * MAX_CHANGE_RATIO) - MAX_CHANGE_RATIO
        
        # 计算预测的收盘价
        last_close = df.iloc[-1]['close']
        pred_close = last_close * (1 + pred_change_ratio)
        
        print(f"🔍 预测变化幅度: {pred_change_ratio*100:.2f}%")
    else:
        # 使用绝对价格预测
        pred_close = inverse_transform_close(pred_normalized)
        last_close = df.iloc[-1]['close']
        pred_change_ratio = (pred_close - last_close) / last_close
        print(f"🔍 原始预测变化幅度: {pred_change_ratio*100:.2f}%")

    direction = "涨" if pred_close > last_close else "跌"

    last_close_time = pd.to_datetime(df.iloc[-1]["timestamp"])
    pred_time = last_close_time + pd.Timedelta(hours=1)

    return {
        "预测收盘价": pred_close,
        "上次收盘价": last_close,
        "预测涨跌": direction,
        "预测时间": pred_time.strftime("%Y-%m-%d %H:%M:%S"),
        "涨跌幅度": f"{((pred_close - last_close) / last_close * 100):.2f}%"
    }

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