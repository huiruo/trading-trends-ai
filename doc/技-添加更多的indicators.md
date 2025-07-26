
## indicators
继续优化特征、模型结构和数据量

## 1. 特征工程优化
a. 增加更多技术指标
动量类：ROC、Stochastic Oscillator、Williams %R
趋势类：SMA（简单均线）、WMA（加权均线）、ADX
成交量类：OBV、MFI
价格结构：K线形态特征（如实体长度、上下影线比例等）

b. 特征交互
计算如 close/open、high/low、(close-open)/open 等比值特征
计算过去N根K线的均值、方差、最大/最小等统计特征

c. 特征归一化
保持所有特征都用同一个scaler归一化，避免信息泄露

## 2. 模型结构优化
a. LSTM/GRU参数调整
增加/减少LSTM层数、隐藏单元数
尝试GRU替代LSTM
增加Dropout防止过拟合

b. 更复杂的模型
堆叠LSTM+全连接层
LSTM+Attention机制
尝试1D卷积（CNN）+LSTM混合结构

c. 损失函数/优化器
尝试不同的优化器（AdamW、RMSprop等）
尝试HuberLoss等更鲁棒的损失函数

## 3. 数据量扩充
a. 获取更多历史K线
尽量获取更长时间段的BTC小时K线（数千~数万条最佳）
可以考虑多品种（ETH、BNB等）联合训练

## b. 数据增强
对K线做微小扰动（如加噪声、平移、缩放）做数据增强
滑动窗口法生成更多训练样本

## 在 technical_indicators.py 
```python
def calculate_sma(data, period):
    return data.rolling(window=period).mean()

def calculate_roc(data, period=12):
    return data.diff(period) / data.shift(period) * 100

def calculate_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)
```

并在 add_technical_indicators 里加上这些新特征。

b. 增加K线结构特征
```python
df['body'] = df['close'] - df['open']
df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
```

## 指标不是越多越好

```
这样我们就有了一个精简版的模型，只使用12个最重要的特征：
基础特征 (5个)：
open, high, low, close, volume
核心指标 (4个)：
ema_12 (指数移动平均线)
rsi_14 (相对强弱指数)
macd (MACD主线)
bb_position (布林带位置)
结构特征 (3个)：
body_ratio (K线实体比例)
close_open_ratio (收盘开盘比)
这个精简版应该能：
减少过拟合
提高训练稳定性
降低计算复杂度
保持核心预测能力
你想要现在重新训练吗？
```

## 分析
```bash
python feature_analysis.py

这个分析工具会：
检查每个特征的统计信息：范围、均值、标准差
检测异常特征：无穷大值、NaN值、数值范围过大
分析特征相关性：找出与价格变化最相关的特征
生成可视化图表：价格分布、相关性热力图等
提供修复建议：哪些特征需要修复或移除
运行后，我们就能知道：
哪些特征有问题（导致极端预测）
哪些特征与价格变化最相关
应该保留哪些特征
这样我们就能有针对性地修复问题特征，而不是盲目地硬编码限制！
```