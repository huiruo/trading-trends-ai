# config_improved.py - 改进版配置

# 模型保存路径
MODEL_PATH = "model/lstm_model_improved.pt"

# 训练窗口大小（增加到48小时）
WINDOW_SIZE = 48

# 基础特征列
BASE_FEATURES = ["open", "high", "low", "close", "volume"]

# 技术指标特征（需要计算）
TECHNICAL_FEATURES = [
    "ema_12", "ema_26", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position"
]

# 所有特征列
FEATURE_COLUMNS = BASE_FEATURES + TECHNICAL_FEATURES

# 预测目标列
TARGET_COLUMN = "close"

# 训练参数
TRAIN_EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# 数据路径
DATA_PATH = "dataset/btc_1h.csv" 