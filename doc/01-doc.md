kline-predictor/
├── app.py                # Flask 接口入口
├── config.py             # 配置文件（如模型路径、超参数）
├── dataset/
│   └── sample.csv        # 示例K线CSV数据（你替换自己的）
├── model/
│   ├── model.py          # PyTorch 模型定义
│   ├── train.py          # 模型训练脚本
│   └── predict.py        # 模型预测脚本
├── preprocess.py         # 数据预处理（归一化、分窗等）
├── requirements.txt      # pip依赖列表
└── README.md             # 项目说明

- preprocess.py
加载 CSV 格式的 K线数据（含 Open/High/Low/Close/Volume）

归一化、滑动窗口切分序列（适用于 LSTM、Transformer 等模型）

- model/model.py
定义一个简单的 LSTM 模型（后续可换成 GRU、Transformer）

- model/train.py
加载预处理后的数据，训练模型并保存权重到 model.pt

- model/predict.py
加载模型和最新数据，进行趋势预测（如下一时刻的收盘价）

- app.py
提供 HTTP 接口 /predict，供前端调用

## 安装依赖-运行
```bash
source ~/.zshrc
python --version
python -m venv venv

➜  trading-trends-ai git:(main) ✗ source venv/bin/activate

(venv) ➜  trading-trends-ai git:(main) ✗ pip install -r requirements.text 

✅ 快速确认是否安装成功

python app.py
```

## http://127.0.0.1:5000
```
(venv) ➜  trading-trends-ai git:(main) ✗ python app.py
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 115-163-768
```

## ✅ dataset/sample.csv
请你手动添加一份 CSV 示例数据，包含至少以下列：

timestamp,open,high,low,close,volume


## 提示词
```
(venv) ➜  trading-trends-ai git:(main) ✗ rm model/scaler.pkl 
rm model/lstm_model_improved.pt
python -m model.train_improved
数据清理完成
CSV 原始K线数据条数: 10159
生成的训练样本序列数: 10111
特征数量: 8
  有问题吗？这是一个预测btc 小时k线项目 ，数据如下_id,close,closeTime,high,low,open,quoteAssetVolume,takerBuyBaseVolume,takerBuyQuoteVolume,trades,updatedAtLocal,volume
1717020000000,67680.01,1717023599999,67826.09,67605.94,67700.42,26754950.9165193,180.04339,12186862.6058975,22438,2025-07-27T03:28:52.034Z,395.25117
1717023600000,67652.42,1717027199999,67731.59,67576.06,67680.01,30721275.7982111,188.3291,12743020.1021687,23042,2025-07-27T03:28:52.034Z,454.0068
```