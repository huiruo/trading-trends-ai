
# 模型定义 model/model.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步
        out = self.fc(out)
        return out

# 检查训练数据中是否有异常值
def analyze_training_data(df):
    # 计算每根K线的涨跌幅
    df['change_ratio'] = df['close'].pct_change()
    print(f"最大涨幅: {df['change_ratio'].max()*100:.2f}%")
    print(f"最大跌幅: {df['change_ratio'].min()*100:.2f}%")
    print(f"平均涨跌幅: {df['change_ratio'].abs().mean()*100:.2f}%")