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

# 训练策略配置
USE_RELATIVE_CHANGE = False  # 使用绝对价格作为训练目标

#  MAX_CHANGE_RATIO 的作用
# 它人为地限制了模型能学到的最大涨跌幅，比如5%。
# 这样做的好处是防止模型输出极端、不现实的预测（比如-10%、+15%）。
# 但副作用是：如果真实市场出现了大于5%的波动，模型永远无法预测出来。
# 当前设置（平衡）
# MAX_CHANGE_RATIO = 0.05     # 最大变化幅度限制 (5%) 

# MAX_CHANGE_RATIO 设得很大（比如0.2或更大），甚至不限制，观察模型真实输出。
# 这样你能看到模型到底有没有“乱跳”。
# MAX_CHANGE_RATIO = 0.2    
MAX_CHANGE_RATIO = 1.0  # 100%变化，实际上就是不限制

# 保守策略（适合稳健交易）：
# MAX_CHANGE_RATIO = 0.03  # 3%

# 激进策略（适合波动大的市场）：
# MAX_CHANGE_RATIO = 0.08  # 8%