## Sat Jul 26 17:43:10 CST 2025
```
(venv) ➜  trading-trends-ai git:(main) ✗ python -m model.train_improved
/Users/ruo/user-workspace/DOCX/trading-trends-ai/technical_indicators.py:55: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  df = df.fillna(method='bfill').fillna(method='ffill')
CSV 原始K线数据条数: 607
生成的训练样本序列数: 559
特征数量: 5
⚠️ No existing model found, training from scratch.
Epoch 1/50, Avg Loss: 0.098118, Best Loss: 0.098118
Epoch 2/50, Avg Loss: 0.017192, Best Loss: 0.017192
Epoch 3/50, Avg Loss: 0.005536, Best Loss: 0.005536
Epoch 4/50, Avg Loss: 0.004185, Best Loss: 0.004185
Epoch 5/50, Avg Loss: 0.003449, Best Loss: 0.003449
Epoch 6/50, Avg Loss: 0.003722, Best Loss: 0.003449
Epoch 7/50, Avg Loss: 0.003578, Best Loss: 0.003449
Epoch 8/50, Avg Loss: 0.003799, Best Loss: 0.003449
Epoch 9/50, Avg Loss: 0.003671, Best Loss: 0.003449
Epoch 10/50, Avg Loss: 0.003115, Best Loss: 0.003115
Epoch 11/50, Avg Loss: 0.003345, Best Loss: 0.003115
Epoch 12/50, Avg Loss: 0.002857, Best Loss: 0.002857
Epoch 13/50, Avg Loss: 0.003233, Best Loss: 0.002857
Epoch 14/50, Avg Loss: 0.002814, Best Loss: 0.002814
Epoch 15/50, Avg Loss: 0.003140, Best Loss: 0.002814
Epoch 16/50, Avg Loss: 0.002856, Best Loss: 0.002814
Epoch 17/50, Avg Loss: 0.003596, Best Loss: 0.002814
Epoch 18/50, Avg Loss: 0.003173, Best Loss: 0.002814
Epoch 19/50, Avg Loss: 0.002464, Best Loss: 0.002464
Epoch 20/50, Avg Loss: 0.002551, Best Loss: 0.002464
Epoch 21/50, Avg Loss: 0.002555, Best Loss: 0.002464
Epoch 22/50, Avg Loss: 0.002781, Best Loss: 0.002464
Epoch 23/50, Avg Loss: 0.002275, Best Loss: 0.002275
Epoch 24/50, Avg Loss: 0.003241, Best Loss: 0.002275
Epoch 25/50, Avg Loss: 0.002264, Best Loss: 0.002264
Epoch 26/50, Avg Loss: 0.002616, Best Loss: 0.002264
Epoch 27/50, Avg Loss: 0.002806, Best Loss: 0.002264
Epoch 28/50, Avg Loss: 0.002223, Best Loss: 0.002223
Epoch 29/50, Avg Loss: 0.002158, Best Loss: 0.002158
Epoch 30/50, Avg Loss: 0.002739, Best Loss: 0.002158
Epoch 31/50, Avg Loss: 0.002395, Best Loss: 0.002158
Epoch 32/50, Avg Loss: 0.002203, Best Loss: 0.002158
Epoch 33/50, Avg Loss: 0.002341, Best Loss: 0.002158
Epoch 34/50, Avg Loss: 0.002073, Best Loss: 0.002073
Epoch 35/50, Avg Loss: 0.002174, Best Loss: 0.002073
Epoch 36/50, Avg Loss: 0.002227, Best Loss: 0.002073
Epoch 37/50, Avg Loss: 0.002118, Best Loss: 0.002073
Epoch 38/50, Avg Loss: 0.002248, Best Loss: 0.002073
Epoch 39/50, Avg Loss: 0.002389, Best Loss: 0.002073
Epoch 40/50, Avg Loss: 0.002674, Best Loss: 0.002073
Epoch 41/50, Avg Loss: 0.002201, Best Loss: 0.002073
Epoch 42/50, Avg Loss: 0.002239, Best Loss: 0.002073
Epoch 43/50, Avg Loss: 0.001899, Best Loss: 0.001899
Epoch 44/50, Avg Loss: 0.002084, Best Loss: 0.001899
Epoch 45/50, Avg Loss: 0.002117, Best Loss: 0.001899
Epoch 46/50, Avg Loss: 0.001961, Best Loss: 0.001899
Epoch 47/50, Avg Loss: 0.002010, Best Loss: 0.001899
Epoch 48/50, Avg Loss: 0.002184, Best Loss: 0.001899
Epoch 49/50, Avg Loss: 0.001958, Best Loss: 0.001899
Epoch 50/50, Avg Loss: 0.001851, Best Loss: 0.001851
✅ Improved model saved to model/lstm_model_improved.pt
```

## 疑问
python -m model.predict_improved 会自动读取 model/lstm_model_improved.pt。
原因如下：
在 model/predict_improved.py 里，模型的加载路径是通过 from config_improved import * 导入的，其中有：
```
MODEL_PATH = "model/lstm_model_improved.pt"

你无需手动指定路径，直接运行即可。
如果你想切换模型，只需要修改 config_improved.py 里的 MODEL_PATH 即可。
```

## 报错
```
ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- bb_lower
- bb_middle
- bb_position
- bb_upper
- bb_width
- ...

执行：
rm model/scaler.pkl
python -m model.train_improved
这会用新特征重新fit scaler，并保存正确的 scaler.pkl。

python -m model.predict_improved

以后每次特征有变化（比如加了技术指标），都要重新训练模型和scaler。
```

## 报错2
```
RuntimeError: Error(s) in loading state_dict for LSTMModel:
	size mismatch for lstm.weight_ih_l0: copying a param with shape torch.Size([512, 5]) from checkpoint, the shape in current model is torch.Size([512, 16]).

只需删除旧的模型权重文件 model/lstm_model_improved.pt，重新训练即可！

   rm model/lstm_model_improved.pt

  python -m model.train_improved
```