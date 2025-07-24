# model/train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from model.model import SimpleLSTM
from preprocess import preprocess
from config import MODEL_PATH, WINDOW_SIZE, FEATURE_COLUMNS

import pandas as pd

def train_model(csv_path):
    df = pd.read_csv(csv_path)
    X, y = preprocess(df)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleLSTM(input_size=X.shape[2])
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for xb, yb in loader:
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_PATH)
