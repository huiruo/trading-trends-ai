# config.py

# 模型保存路径（和 train.py/predict.py 一致）
MODEL_PATH = "model/lstm_model.pt"

# 训练窗口大小（即时间序列长度）
# 例如用前48小时预测下一个小时
WINDOW_SIZE = 4

# K线输入特征列（可以是 open/high/low/close/volume 等）
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]

# 预测目标列（例如预测 close 收盘价）
TARGET_COLUMN = "close"

# CSV 数据路径（相对于根目录）
DATA_PATH = "dataset/btc_1h.csv"
