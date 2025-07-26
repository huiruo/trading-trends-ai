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
import numpy as np

def train_improved(csv_path: str, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    # 加载数据
    df = load_and_preprocess(csv_path)
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 检查数据是否有NaN或无穷大
    print("检查数据质量...")
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"NaN值数量: {nan_count}")
    print(f"无穷大值数量: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("⚠️ 发现NaN或无穷大值，正在清理...")
        # 用前向填充和后向填充清理NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        # 用0替换无穷大值
        df = df.replace([np.inf, -np.inf], 0)
        print("数据清理完成")
    
    # 创建序列
    X, y = create_sequences(df, window_size=WINDOW_SIZE)
    
    print(f"CSV 原始K线数据条数: {len(df)}")
    print(f"生成的训练样本序列数: {len(X)}")
    print(f"特征数量: {X.shape[2]}")
    
    # 检查训练数据
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X min: {np.min(X)}, X max: {np.max(X)}")
    print(f"y min: {np.min(y)}, y max: {np.max(y)}")
    print(f"X中有NaN: {np.isnan(X).any()}")
    print(f"y中有NaN: {np.isnan(y).any()}")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建改进的模型
    model = LSTMModel(input_size=X.shape[2], hidden_size=16, num_layers=1)

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
        batch_count = 0
        for xb, yb in dataloader:
            # 检查批次数据
            if torch.isnan(xb).any() or torch.isnan(yb).any():
                print(f"⚠️ Epoch {epoch+1}: 发现NaN值在批次数据中")
                continue
                
            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            # 检查loss是否为NaN
            if torch.isnan(loss):
                print(f"⚠️ Epoch {epoch+1}: Loss为NaN，跳过此批次")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                # 保存最佳模型
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
            
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}, Best Loss: {best_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}: 所有批次都包含NaN，跳过此epoch")

    print(f"✅ Improved model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_improved(DATA_PATH) 