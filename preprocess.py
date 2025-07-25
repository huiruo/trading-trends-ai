# preprocess.py
# 用于读取 CSV、标准化特征、构造时间窗口等处理：
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import WINDOW_SIZE, FEATURE_COLUMNS, TARGET_COLUMN

scaler = MinMaxScaler()

def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
    return df

def create_sequences(df: pd.DataFrame, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(df) - window_size):
        x_seq = df[FEATURE_COLUMNS].iloc[i:i+window_size].values
        y_target = df[TARGET_COLUMN].iloc[i + window_size]
        X.append(x_seq)
        y.append(y_target)
    return np.array(X), np.array(y)

def inverse_transform(value: float):
    # Inverse transform only the 'close' column (assumed to be 4th column)
    dummy = np.zeros((1, len(FEATURE_COLUMNS)))
    dummy[0, FEATURE_COLUMNS.index(TARGET_COLUMN)] = value
    return scaler.inverse_transform(dummy)[0, FEATURE_COLUMNS.index(TARGET_COLUMN)]