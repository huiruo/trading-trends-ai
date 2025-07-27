# script_evaluate_model.py - 统一模型评估脚本
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
from preprocess import load_and_preprocess, create_sequences, load_scaler
from technical_indicators import add_technical_indicators
from model.model import LSTMModel
from config_improved import *

def evaluate_model():
    """评估模型在训练集上的表现"""
    
    print("=== 模型评估开始 ===")
    print(f"模型类型: {'分类' if USE_CLASSIFICATION else '回归'}")
    
    # 加载数据
    df = load_and_preprocess(DATA_PATH)
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
    X, _ = create_sequences(df, window_size=WINDOW_SIZE)
    
    print(f"训练样本数: {len(X)}")
    print(f"特征数量: {X.shape[2]}")
    
    if USE_CLASSIFICATION:
        evaluate_classification_model(X, df)
    else:
        evaluate_regression_model(X, df)

def evaluate_classification_model(X: np.ndarray, df: pd.DataFrame):
    """评估分类模型"""
    print("\n=== 分类模型评估 ===")
    
    # 加载分类模型
    model = LSTMModel(
        input_size=X.shape[2], 
        hidden_size=64, 
        num_layers=2, 
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    # 预测训练集
    predictions = []
    actual_changes = []
    direction_accuracy = 0
    total_predictions = 0
    class_predictions = []
    class_actuals = []
    
    with torch.no_grad():
        for i in range(len(X)):
            x_tensor = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)
            pred_probs = model(x_tensor)
            pred_class = torch.argmax(pred_probs, dim=1).item()
            confidence = torch.max(pred_probs, dim=1).values.item()
            
            # 根据分类结果计算预测价格变化
            current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
            if pred_class == 0:  # 跌
                # 根据置信度调整预测幅度
                pred_change = -0.002 * confidence  # 0.2% * 置信度
                pred_close = current_close * (1 + pred_change)
                pred_direction = "跌"
            else:  # 涨
                # 根据置信度调整预测幅度
                pred_change = 0.002 * confidence  # 0.2% * 置信度
                pred_close = current_close * (1 + pred_change)
                pred_direction = "涨"
            
            pred_change_ratio = (pred_close - current_close) / current_close
            
            # 实际变化
            actual_close = df.iloc[i + WINDOW_SIZE]['close']
            actual_change_ratio = (actual_close - current_close) / current_close
            
            # 实际方向
            if actual_change_ratio < 0:
                actual_direction = "跌"
                actual_class = 0
            else:
                actual_direction = "涨"
                actual_class = 1
            
            # 方向准确率（二元分类）
            if pred_direction == actual_direction:
                direction_accuracy += 1
            total_predictions += 1
            
            # 分类准确率
            if pred_class == actual_class:
                class_accuracy = 1
            else:
                class_accuracy = 0
            
            predictions.append(pred_change_ratio)
            actual_changes.append(actual_change_ratio)
            class_predictions.append(pred_class)
            class_actuals.append(actual_class)
    
    # 计算统计信息
    predictions = np.array(predictions)
    actual_changes = np.array(actual_changes)
    class_predictions = np.array(class_predictions)
    class_actuals = np.array(class_actuals)
    
    print(f"\n=== 分类模型预测统计 ===")
    print(f"方向准确率: {direction_accuracy/total_predictions*100:.2f}%")
    print(f"分类准确率 (全部): {np.mean(class_predictions == class_actuals)*100:.2f}%")
    print(f"预测变化范围: [{predictions.min()*100:.3f}%, {predictions.max()*100:.3f}%]")
    print(f"实际变化范围: [{actual_changes.min()*100:.3f}%, {actual_changes.max()*100:.3f}%]")
    print(f"预测变化均值: {predictions.mean()*100:.3f}%")
    print(f"实际变化均值: {actual_changes.mean()*100:.3f}%")
    
    # 分类混淆矩阵
    print(f"\n=== 分类混淆矩阵 ===")
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(class_actuals, class_predictions)
    print("混淆矩阵 (行=实际, 列=预测):")
    print("     跌  涨")
    for i, label in enumerate(['跌', '涨']):
        print(f"{label}: {cm[i]}")
    
    # 详细分类报告
    print(f"\n=== 详细分类报告 ===")
    print(classification_report(class_actuals, class_predictions, 
                               target_names=['跌', '涨'], 
                               zero_division=0))
    
    # 分析最后几个预测
    print(f"\n=== 最后10个预测分析 ===")
    for i in range(max(0, len(predictions)-10), len(predictions)):
        pred_dir = "涨" if predictions[i] > 0 else "跌"
        actual_dir = "涨" if actual_changes[i] > 0 else "跌"
        correct = "✅" if pred_dir == actual_dir else "❌"
        
        # 获取当前收盘价和下一期收盘价
        current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
        next_close = df.iloc[i + WINDOW_SIZE]['close']
        
        print(f"预测{i+1}: {pred_dir} {predictions[i]*100:.3f}% | "
              f"当前价: {current_close:.2f} | "
              f"实际: {actual_dir} {actual_changes[i]*100:.3f}% | "
              f"下期价: {next_close:.2f} {correct}")
    
    # 分析预测偏差
    bias = predictions.mean() - actual_changes.mean()
    print(f"\n预测偏差: {bias*100:.3f}% (正值表示预测偏乐观，负值表示预测偏悲观)")

def evaluate_regression_model(X: np.ndarray, df: pd.DataFrame):
    """评估回归模型"""
    print("\n=== 回归模型评估 ===")
    
    # 加载回归模型
    model = LSTMModel(
        input_size=X.shape[2], 
        hidden_size=64, 
        num_layers=2, 
        num_classes=1
    )
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
            
            # 将归一化的预测值转换回实际变化率
            pred_change_ratio = (pred_normalized * 2 * MAX_CHANGE_RATIO) - MAX_CHANGE_RATIO
            
            # 实际变化
            current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
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
    
    # 计算MSE和MAE
    mse = np.mean((predictions - actual_changes) ** 2)
    mae = np.mean(np.abs(predictions - actual_changes))
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    
    # 计算R²
    ss_res = np.sum((predictions - actual_changes) ** 2)
    ss_tot = np.sum((actual_changes - actual_changes.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 分析最后几个预测
    print(f"\n=== 最后10个预测分析 ===")
    for i in range(max(0, len(predictions)-10), len(predictions)):
        pred_dir = "涨" if predictions[i] > 0 else "跌"
        actual_dir = "涨" if actual_changes[i] > 0 else "跌"
        correct = "✅" if pred_dir == actual_dir else "❌"
        
        # 获取当前收盘价和下一期收盘价
        current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
        next_close = df.iloc[i + WINDOW_SIZE]['close']
        
        print(f"预测{i+1}: {pred_dir} {predictions[i]*100:.2f}% | "
              f"当前价: {current_close:.2f} | "
              f"实际: {actual_dir} {actual_changes[i]*100:.2f}% | "
              f"下期价: {next_close:.2f} {correct}")
    
    # 分析预测偏差
    bias = predictions.mean() - actual_changes.mean()
    print(f"\n预测偏差: {bias*100:.2f}% (正值表示预测偏乐观，负值表示预测偏悲观)")

if __name__ == "__main__":
    evaluate_model() 