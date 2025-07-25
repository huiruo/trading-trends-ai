# 训练脚本 model/train.py
# 用于训练模型并保存权重：
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_and_preprocess, create_sequences
from model.model import LSTMModel
from config import MODEL_PATH, WINDOW_SIZE
import os

def train(csv_path: str, epochs=20, lr=0.001, batch_size=64):
    df = load_and_preprocess(csv_path)
    X, y = create_sequences(df)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_size=X.shape[2])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for xb, yb in dataloader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
