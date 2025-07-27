# evaluate_model.py
# python script_evaluate_model.py
# 这个脚本会告诉我们：
# 方向准确率：模型预测涨跌方向的准确率
# 预测偏差：模型是否总是偏乐观或偏悲观
# 最后几个预测：看看模型在训练集末尾的表现
# 这样我们就能知道：
# 模型是否真的学会了预测
# 预测错误是偶然的还是系统性的
# 需要如何改进模型
import pandas as pd
import numpy as np
import torch
from preprocess import load_and_preprocess, create_sequences, load_scaler, inverse_transform_close
from technical_indicators import add_technical_indicators
from model.model import LSTMModel
from config_improved import *

def evaluate_model():
    """评估模型在训练集上的表现"""
    
    # 加载数据
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={
        'closeTime': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 创建序列
    X, y = create_sequences(df, window_size=WINDOW_SIZE)
    
    print("=== 模型评估报告 ===")
    print(f"训练样本数: {len(X)}")
    print(f"特征数量: {X.shape[2]}")
    
    # 检查是否使用分类方法
    try:
        from config_improved import USE_CLASSIFICATION, CLASSIFICATION_THRESHOLD
    except ImportError:
        USE_CLASSIFICATION = False
        CLASSIFICATION_THRESHOLD = 0.001
    
    if USE_CLASSIFICATION:
        print("使用分类模型评估")
        # 加载分类模型
        model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=2, num_classes=3)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        # 预测训练集
        predictions = []
        actual_changes = []
        direction_accuracy = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i in range(len(X)):
                x_tensor = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)
                pred_probs = model(x_tensor)
                pred_class = torch.argmax(pred_probs, dim=1).item()
                
                # 根据分类结果计算预测价格变化
                current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
                if pred_class == 0:  # 跌
                    pred_close = current_close * 0.999
                    pred_direction = "跌"
                elif pred_class == 2:  # 涨
                    pred_close = current_close * 1.001
                    pred_direction = "涨"
                else:  # 平
                    pred_close = current_close
                    pred_direction = "平"
                
                pred_change_ratio = (pred_close - current_close) / current_close
                
                # 实际变化
                actual_close = df.iloc[i + WINDOW_SIZE]['close']
                actual_change_ratio = (actual_close - current_close) / current_close
                
                # 实际方向
                if actual_change_ratio < -CLASSIFICATION_THRESHOLD:
                    actual_direction = "跌"
                elif actual_change_ratio > CLASSIFICATION_THRESHOLD:
                    actual_direction = "涨"
                else:
                    actual_direction = "平"
                
                # 方向准确率（只考虑涨跌，不考虑平）
                if pred_direction != "平" and actual_direction != "平":
                    if pred_direction == actual_direction:
                        direction_accuracy += 1
                    total_predictions += 1
                
                predictions.append(pred_change_ratio)
                actual_changes.append(actual_change_ratio)
        
        # 计算统计信息
        predictions = np.array(predictions)
        actual_changes = np.array(actual_changes)
        
        print(f"\n=== 分类模型预测统计 ===")
        if total_predictions > 0:
            print(f"方向准确率: {direction_accuracy/total_predictions*100:.2f}%")
        else:
            print("方向准确率: 无有效预测")
        print(f"预测变化范围: [{predictions.min()*100:.3f}%, {predictions.max()*100:.3f}%]")
        print(f"实际变化范围: [{actual_changes.min()*100:.3f}%, {actual_changes.max()*100:.3f}%]")
        print(f"预测变化均值: {predictions.mean()*100:.3f}%")
        print(f"实际变化均值: {actual_changes.mean()*100:.3f}%")
        
        # 分析最后几个预测
        print(f"\n=== 最后5个预测分析 ===")
        for i in range(max(0, len(predictions)-5), len(predictions)):
            pred_dir = "涨" if predictions[i] > 0 else "跌" if predictions[i] < 0 else "平"
            actual_dir = "涨" if actual_changes[i] > 0 else "跌" if actual_changes[i] < 0 else "平"
            correct = "✅" if pred_dir == actual_dir else "❌"
            
            print(f"预测{i+1}: {pred_dir} {predictions[i]*100:.3f}% | 实际: {actual_dir} {actual_changes[i]*100:.3f}% {correct}")
        
        # 分析预测偏差
        bias = predictions.mean() - actual_changes.mean()
        print(f"\n预测偏差: {bias*100:.3f}% (正值表示预测偏乐观，负值表示预测偏悲观)")
        
    else:
        print("使用回归模型评估")
        # 加载回归模型
        model = LSTMModel(input_size=X.shape[2], hidden_size=64, num_layers=2, num_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        # 预测训练集
        predictions = []
        actual_changes = []
        direction_accuracy = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i in range(len(X)):
                x_tensor = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)
                pred_normalized = model(x_tensor).item()
                
                # 获取实际值
                try:
                    from config_improved import USE_RELATIVE_CHANGE, MAX_CHANGE_RATIO
                except ImportError:
                    USE_RELATIVE_CHANGE = False
                    MAX_CHANGE_RATIO = 0.05
                
                if USE_RELATIVE_CHANGE:
                    # 相对变化模式
                    pred_change_ratio = (pred_normalized * 2 * MAX_CHANGE_RATIO) - MAX_CHANGE_RATIO
                    current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
                    pred_close = current_close * (1 + pred_change_ratio)
                else:
                    # 绝对价格模式
                    pred_close = inverse_transform_close(pred_normalized)
                    current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
                    pred_change_ratio = (pred_close - current_close) / current_close
                
                # 实际变化
                actual_close = df.iloc[i + WINDOW_SIZE]['close']
                actual_change_ratio = (actual_close - current_close) / current_close
                
                # 方向预测
                pred_direction = "涨" if pred_change_ratio > 0 else "跌"
                actual_direction = "涨" if actual_change_ratio > 0 else "跌"
                
                if pred_direction == actual_direction:
                    direction_accuracy += 1
                total_predictions += 1
                
                predictions.append(pred_change_ratio)
                actual_changes.append(actual_change_ratio)
        
        # 计算统计信息
        predictions = np.array(predictions)
        actual_changes = np.array(actual_changes)
        
        print(f"\n=== 回归模型预测统计 ===")
        print(f"方向准确率: {direction_accuracy/total_predictions*100:.2f}%")
        print(f"预测变化范围: [{predictions.min()*100:.2f}%, {predictions.max()*100:.2f}%]")
        print(f"实际变化范围: [{actual_changes.min()*100:.2f}%, {actual_changes.max()*100:.2f}%]")
        print(f"预测变化均值: {predictions.mean()*100:.2f}%")
        print(f"实际变化均值: {actual_changes.mean()*100:.2f}%")
        print(f"预测变化标准差: {predictions.std()*100:.2f}%")
        print(f"实际变化标准差: {actual_changes.std()*100:.2f}%")
        
        # 计算MSE
        mse = np.mean((predictions - actual_changes) ** 2)
        print(f"均方误差 (MSE): {mse:.6f}")
        
        # 分析最后几个预测
        print(f"\n=== 最后5个预测分析 ===")
        for i in range(max(0, len(predictions)-5), len(predictions)):
            pred_dir = "涨" if predictions[i] > 0 else "跌"
            actual_dir = "涨" if actual_changes[i] > 0 else "跌"
            correct = "✅" if pred_dir == actual_dir else "❌"
            
            print(f"预测{i+1}: {pred_dir} {predictions[i]*100:.2f}% | 实际: {actual_dir} {actual_changes[i]*100:.2f}% {correct}")
        
        # 分析预测偏差
        bias = predictions.mean() - actual_changes.mean()
        print(f"\n预测偏差: {bias*100:.2f}% (正值表示预测偏乐观，负值表示预测偏悲观)")

if __name__ == "__main__":
    evaluate_model() 