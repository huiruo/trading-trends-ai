# model/predict.py
import torch
import pandas as pd
from model.model import SimpleLSTM
from preprocess import preprocess
from config import MODEL_PATH

def predict_from_csv(file):
    df = pd.read_csv(file)
    X, _ = preprocess(df)
    X = torch.tensor(X, dtype=torch.float32)

    model = SimpleLSTM(input_size=X.shape[2])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        preds = model(X).squeeze().numpy().tolist()
    return {"predictions": preds}
