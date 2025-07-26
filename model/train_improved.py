# model/train_improved.py - 改进版训练脚本
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_and_preprocess, create_sequences
from technical_indicators import add_technical_indicators
from model.model import LSTMModel
from config_improved import *

def train_improved(csv_path: str, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    # 加载数据
    df = load_and_preprocess(csv_path)
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 创建序列
    X, y = create_sequences(df, window_size=WINDOW_SIZE)
    
    print(f"CSV 原始K线数据条数: {len(df)}")
    print(f"生成的训练样本序列数: {len(X)}")
    print(f"特征数量: {X.shape[2]}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建改进的模型
    model = LSTMModel(input_size=X.shape[2], hidden_size=128, num_layers=3)

    # 加载已有权重（如果存在）
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Loading existing model weights from {MODEL_PATH} for continued training.")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("⚠️ No existing model found, training from scratch.")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = float('inf')
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
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存最佳模型
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}, Best Loss: {best_loss:.6f}")

    print(f"✅ Improved model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_improved(DATA_PATH) 