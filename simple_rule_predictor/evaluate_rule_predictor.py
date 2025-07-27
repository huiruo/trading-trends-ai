# evaluate_rule_predictor.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from simple_rule_predictor import calculate_simple_prediction
from config_improved import DATA_PATH

# python simple_rule_predictor/evaluate_rule_predictor.py
def evaluate_rule_predictor():
    """评估规则预测器的准确率"""
    
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
    
    print("=== 规则预测器评估报告 ===")
    print(f"总数据条数: {len(df)}")
    
    # 从第100根K线开始评估（确保有足够的历史数据计算技术指标）
    start_idx = 100
    end_idx = len(df) - 1
    
    predictions = []
    actual_changes = []
    direction_accuracy = 0
    total_predictions = 0
    
    print(f"评估范围: 第{start_idx}根K线到第{end_idx}根K线")
    print(f"评估样本数: {end_idx - start_idx}")
    print("正在评估中...")
    
    for i in range(start_idx, end_idx):
        # 使用到第i根K线的数据做预测
        df_subset = df.iloc[:i+1].copy()
        
        try:
            # 预测下一根K线（静默模式）
            result = calculate_simple_prediction(df_subset, silent=True)
            
            # 获取实际下一根K线的数据
            actual_close = df.iloc[i+1]['close']
            current_close = df.iloc[i]['close']
            actual_change_ratio = (actual_close - current_close) / current_close
            
            # 记录预测和实际结果
            pred_direction = result['预测涨跌']
            pred_change_ratio = float(result['涨跌幅度'].replace('%', '')) / 100
            
            # 确定实际方向
            if actual_change_ratio > 0.001:  # 0.1%以上算涨
                actual_direction = "涨"
            elif actual_change_ratio < -0.001:  # -0.1%以下算跌
                actual_direction = "跌"
            else:
                actual_direction = "平"
            
            # 计算方向准确率
            if pred_direction == actual_direction:
                direction_accuracy += 1
            total_predictions += 1
            
            predictions.append(pred_change_ratio)
            actual_changes.append(actual_change_ratio)
                
        except Exception as e:
            continue
    
    # 计算统计信息
    if total_predictions > 0:
        predictions = np.array(predictions)
        actual_changes = np.array(actual_changes)
        
        print(f"\n=== 规则预测器统计 ===")
        print(f"总预测次数: {total_predictions}")
        print(f"方向准确率: {direction_accuracy/total_predictions*100:.2f}%")
        print(f"预测变化范围: [{predictions.min()*100:.3f}%, {predictions.max()*100:.3f}%]")
        print(f"实际变化范围: [{actual_changes.min()*100:.3f}%, {actual_changes.max()*100:.3f}%]")
        print(f"预测变化均值: {predictions.mean()*100:.3f}%")
        print(f"实际变化均值: {actual_changes.mean()*100:.3f}%")
        
        # 计算MSE
        mse = np.mean((predictions - actual_changes) ** 2)
        print(f"均方误差 (MSE): {mse:.6f}")
        
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
        
        # 按置信度分析准确率
        print(f"\n=== 按置信度分析 ===")
        # 这里可以进一步分析不同置信度下的准确率
        
    else:
        print("没有有效的预测结果")

if __name__ == "__main__":
    evaluate_rule_predictor() 