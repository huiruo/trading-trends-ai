# config_improved.py
import os

# 模型配置
MODEL_PATH = "model/lstm_model_improved.pt"
WINDOW_SIZE = 24  # 减少窗口大小，避免过拟合
TRAIN_EPOCHS = 80  # 减少训练轮数
LEARNING_RATE = 0.001  # 提高学习率
BATCH_SIZE = 64  # 适中的批次大小

# 数据路径
# DATA_PATH = "dataset/btc_1h.csv"
# DATA_PATH = "dataset/btc_1hB.csv"
# DATA_PATH = "dataset/btc_1hC.csv"
DATA_PATH = "dataset/btc_1h_sorted.csv"

# 基础特征 - 只保留最核心的OHLCV
BASE_FEATURES = ['open', 'high', 'low', 'close', 'volume']

# 技术指标特征 - 简化特征组合
TECHNICAL_FEATURES = [
    'rsi_14',         # RSI指标 (超买超卖)
    'bb_position',    # 布林带位置 (价格位置)
    'close_open_ratio', # 收盘开盘比 (K线形态)
    'macd_histogram', # MACD柱状图 (趋势)
    'ma5_ratio',      # 5日均线比率 (短期趋势)
    'volume_ratio_5'  # 5日成交量比率 (成交量)
]

# 合并所有特征 - 总共11个特征
FEATURE_COLUMNS = BASE_FEATURES + TECHNICAL_FEATURES

# 训练策略配置 - 使用简单的二分类
USE_CLASSIFICATION = True  # 使用分类方法
USE_RELATIVE_CHANGE = False  # 不使用相对变化

# 分类阈值 - 使用更严格的阈值
CLASSIFICATION_THRESHOLD = 0.002  # 0.2%的变化作为涨跌判断阈值

# 回归配置
MAX_CHANGE_RATIO = 0.02  # 2%的最大变化幅度限制，更现实的范围