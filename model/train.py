# 训练脚本 model/train.py
# 用于训练模型并保存权重：
# model/train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_and_preprocess, create_sequences
from model.model import LSTMModel
from config import MODEL_PATH, WINDOW_SIZE
from config import DATA_PATH

def train(csv_path: str, epochs=20, lr=0.001, batch_size=64):
    df = load_and_preprocess(csv_path)
    X, y = create_sequences(df, window_size=WINDOW_SIZE)
    
    print(f"CSV 原始K线数据条数: {len(df)}")
    print(f"生成的训练样本序列数: {len(X)}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_size=X.shape[2])

    # 加载已有权重（如果存在）
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Loading existing model weights from {MODEL_PATH} for continued training.")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("⚠️ No existing model found, training from scratch.")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    # 默认从配置文件指定的数据路径训练
    train(DATA_PATH)
