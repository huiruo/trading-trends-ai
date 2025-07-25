# model/predict.py
# 用于加载模型并进行下一时刻预测：
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from model.model import LSTMModel
# from preprocess import preprocess, create_sequences
from preprocess import load_and_preprocess, create_sequences
from config import MODEL_PATH, WINDOW_SIZE

def predict_next_candle_direction(df: pd.DataFrame):
    df_processed = load_and_preprocess(df)

    X, y = create_sequences(df_processed, window_size=WINDOW_SIZE)

    X_latest = X[-1]
    X_tensor = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0)

    model = LSTMModel(input_size=X_tensor.shape[2])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        pred_close = model(X_tensor).item()

    last_close = y[-1]

    direction = "涨" if pred_close > last_close else "跌"

    last_close_time = pd.to_datetime(df.iloc[-1]["closeTime"], unit='ms')
    pred_time = last_close_time + pd.Timedelta(hours=1)

    return {
        "预测收盘价": pred_close,
        "上次收盘价": last_close,
        "预测涨跌": direction,
        "预测时间": pred_time.strftime("%Y-%m-%d %H:%M:%S"),
    }

if __name__ == "__main__":
    df = pd.read_csv("dataset/btc_1h.csv")
    result = predict_next_candle_direction(df)
    print(result)