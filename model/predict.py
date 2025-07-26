# model/predict.py
# 用于加载模型并进行下一时刻预测：
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from model.model import LSTMModel
from preprocess import load_and_preprocess, create_sequences, load_scaler, inverse_transform_close
from config import MODEL_PATH, WINDOW_SIZE

def predict_next_candle_direction(df: pd.DataFrame):
    # 加载已保存的scaler
    loaded_scaler = load_scaler()
    if loaded_scaler is None:
        print("⚠️ No scaler found. Please train the model first.")
        return None
    
    # 使用已保存的scaler进行标准化
    df_processed = df.copy()
    df_processed[['open', 'high', 'low', 'close', 'volume']] = loaded_scaler.transform(df[['open', 'high', 'low', 'close', 'volume']])

    X, y = create_sequences(df_processed, window_size=WINDOW_SIZE)
    
    if len(X) == 0:
        print("⚠️ Not enough data for prediction. Need at least {} data points.".format(WINDOW_SIZE + 1))
        return None

    X_latest = X[-1]
    X_tensor = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0)

    model = LSTMModel(input_size=X_tensor.shape[2])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        pred_close_normalized = model(X_tensor).item()

    # 反向转换预测的收盘价
    pred_close = inverse_transform_close(pred_close_normalized)
    
    # 获取实际的最后收盘价
    last_close = df.iloc[-1]['close']

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
    from config import DATA_PATH
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
    result = predict_next_candle_direction(df)
    if result:
        print("\n【预测的下一根K线】")
        print(result)
        # 如果数据集里有真实的下一根K线（比如你有未来数据），也打印出来
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