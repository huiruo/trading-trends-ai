# preprocess.py
import pandas as pd
import numpy as np
from config import FEATURE_COLUMNS, TARGET_COLUMN, WINDOW_SIZE

def preprocess(df: pd.DataFrame):
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    df = (df - df.mean()) / df.std()

    X, y = [], []
    for i in range(len(df) - WINDOW_SIZE):
        X.append(df.iloc[i:i+WINDOW_SIZE][FEATURE_COLUMNS].values)
        y.append(df.iloc[i+WINDOW_SIZE][TARGET_COLUMN])
    return np.array(X), np.array(y)
