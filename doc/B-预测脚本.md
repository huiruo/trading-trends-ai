
2.1 预测脚本 model/predict.py
当前模型输出是预测收盘价（连续值），你想要涨跌分类，可以新增一段涨跌判断逻辑

需要取最新一条完整窗口数据（比如最新4条）作为输入，预测下一小时收盘价

根据预测收盘价和最新收盘价比对，判断涨跌

返回预测涨跌和预测时间

假设你已经有最新的csv文件（比如 btc_1h.csv）里面是连续的K线数据，你想基于最新连续4条做预测：


## 执行
```bash
# 优先用这个
python -m model.predict_improved
python -m model.predict

python model/predict.py
```