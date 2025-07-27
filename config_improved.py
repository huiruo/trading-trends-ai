# config_improved.py
import os

# 模型配置
MODEL_PATH = "model/lstm_model_improved.pt"
WINDOW_SIZE = 24  # 减少窗口大小，避免过拟合
TRAIN_EPOCHS = 80  # 减少训练轮数
LEARNING_RATE = 0.001  # 提高学习率
BATCH_SIZE = 64  # 适中的批次大小

# 数据路径
DATA_PATH = "dataset/btc_1h_sorted.csv"

# ===== 模型类型配置 =====
# 重要：只能选择一种模式，不能混用
USE_CLASSIFICATION = True  # True=分类模型，False=回归模型

# ===== 分类模型配置 =====
CLASSIFICATION_THRESHOLD = 0.0005  # 降低阈值到0.05%，更敏感
NUM_CLASSES = 2  # 2分类：跌、涨 (移除平盘)

# ===== 回归模型配置 =====
MAX_CHANGE_RATIO = 0.02  # 2%的最大变化幅度限制
USE_RELATIVE_CHANGE = True  # 使用相对变化率而不是绝对价格

# ===== 特征工程配置 =====
# 基础价格特征 - 使用对数收益率等平稳特征
BASE_FEATURES = [
    'log_return',           # 对数收益率 (平稳)
    'high_low_ratio',       # 高低价比率 (平稳)
    'volume_log_return',    # 成交量对数收益率 (平稳)
    'price_position'        # 价格在当日区间的位置 (平稳)
]

# 技术指标特征 - 使用z-score标准化，减少冗余
TECHNICAL_FEATURES = [
    'rsi_14_zscore',        # RSI的z-score (标准化)
    'bb_position',          # 布林带位置 (0-1范围)
    'macd_histogram_zscore', # MACD柱状图的z-score
    'ma_cross_signal',      # 均线交叉信号 (-1,0,1)
    'volume_ma_ratio',      # 成交量与均线比率
    'momentum_5_zscore'     # 5日动量的z-score
]

# 合并所有特征
FEATURE_COLUMNS = BASE_FEATURES + TECHNICAL_FEATURES

# ===== 数据预处理配置 =====
USE_ZSCORE_NORMALIZATION = True  # 使用z-score标准化而不是MinMax
ROLLING_WINDOW = 100  # z-score计算的滚动窗口