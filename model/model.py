
# 模型定义 model/model.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze(-1)

# 检查训练数据中是否有异常值
def analyze_training_data(df):
    # 计算每根K线的涨跌幅
    df['change_ratio'] = df['close'].pct_change()
    print(f"最大涨幅: {df['change_ratio'].max()*100:.2f}%")
    print(f"最大跌幅: {df['change_ratio'].min()*100:.2f}%")
    print(f"平均涨跌幅: {df['change_ratio'].abs().mean()*100:.2f}%")