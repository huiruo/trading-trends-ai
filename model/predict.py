# model/predict.py
# 用于加载模型并进行下一时刻预测：
import torch
import numpy as np
from model.model import LSTMModel
from config import MODEL_PATH, WINDOW_SIZE, FEATURE_COLUMNS
from preprocess import load_and_preprocess, inverse_transform

def predict_next(csv_path: str):
    df = load_and_preprocess(csv_path)
    latest_seq = df[FEATURE_COLUMNS].iloc[-WINDOW_SIZE:].values
    x_input = torch.tensor(latest_seq, dtype=torch.float32).unsqueeze(0)

    model = LSTMModel(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        prediction = model(x_input).item()
        prediction_original = inverse_transform(prediction)
        return prediction_original
