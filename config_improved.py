# config_improved.py
import os

# 模型配置
MODEL_PATH = "model/lstm_model_improved.pt"
WINDOW_SIZE = 48
TRAIN_EPOCHS = 50
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

# 数据路径
DATA_PATH = "dataset/btc_1h.csv"

# 基础特征 - 只保留最核心的OHLCV
BASE_FEATURES = ['open', 'high', 'low', 'close', 'volume']

# 技术指标特征 - 只保留最常用、最核心的指标
TECHNICAL_FEATURES = [
    'rsi_14',         # RSI指标 (最常用的超买超卖指标)
    'bb_position',    # 布林带位置 (价格在布林带中的位置)
    'close_open_ratio' # 收盘开盘比 (K线涨跌幅度)
]

# 合并所有特征 - 总共只有8个特征
FEATURE_COLUMNS = BASE_FEATURES + TECHNICAL_FEATURES 