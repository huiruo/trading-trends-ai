# evaluate_hybrid_predictor.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from hybrid_predictor import calculate_hybrid_prediction
from config_improved import DATA_PATH

def evaluate_hybrid_predictor():
    """评估混合预测器的准确率"""
    
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
    
    print("=== 混合预测器评估报告 ===")
    print(f"总数据条数: {len(df)}")
    
    # 从第100根K线开始评估（确保有足够的历史数据计算技术指标）
    start_idx = 100
    end_idx = len(df) - 1
    
    predictions = []
    actual_changes = []
    direction_accuracy = 0
    total_predictions = 0
    
    # 按置信度统计
    high_confidence_correct = 0
    high_confidence_total = 0
    medium_confidence_correct = 0
    medium_confidence_total = 0
    low_confidence_correct = 0
    low_confidence_total = 0
    
    # 按综合得分统计
    score_ranges = {
        'strong_bullish': {'correct': 0, 'total': 0, 'range': (1.0, float('inf'))},
        'moderate_bullish': {'correct': 0, 'total': 0, 'range': (0.5, 1.0)},
        'weak_bullish': {'correct': 0, 'total': 0, 'range': (0.1, 0.5)},
        'neutral': {'correct': 0, 'total': 0, 'range': (-0.1, 0.1)},
        'weak_bearish': {'correct': 0, 'total': 0, 'range': (-0.5, -0.1)},
        'moderate_bearish': {'correct': 0, 'total': 0, 'range': (-1.0, -0.5)},
        'strong_bearish': {'correct': 0, 'total': 0, 'range': (-float('inf'), -1.0)}
    }
    
    print(f"评估范围: 第{start_idx}根K线到第{end_idx}根K线")
    print(f"评估样本数: {end_idx - start_idx}")
    print("正在评估中...")
    
    for i in range(start_idx, end_idx):
        # 使用到第i根K线的数据做预测
        df_subset = df.iloc[:i+1].copy()
        
        try:
            # 预测下一根K线（静默模式）
            result = calculate_hybrid_prediction(df_subset, silent=True)
            
            # 获取实际下一根K线的数据
            actual_close = df.iloc[i+1]['close']
            current_close = df.iloc[i]['close']
            actual_change_ratio = (actual_close - current_close) / current_close
            
            # 记录预测和实际结果
            pred_direction = result['预测涨跌']
            pred_change_ratio = float(result['涨跌幅度'].replace('%', '')) / 100
            confidence = result['置信度']
            total_score = result['综合得分']
            
            # 确定实际方向（使用更宽松的标准）
            if actual_change_ratio > 0.0005:  # 0.05%以上算涨
                actual_direction = "涨"
            elif actual_change_ratio < -0.0005:  # -0.05%以下算跌
                actual_direction = "跌"
            else:
                actual_direction = "平"
            
            # 计算方向准确率
            if pred_direction == actual_direction:
                direction_accuracy += 1
                
                # 按置信度统计准确率
                if confidence == "高":
                    high_confidence_correct += 1
                elif confidence == "中":
                    medium_confidence_correct += 1
                else:
                    low_confidence_correct += 1
            
            total_predictions += 1
            
            # 按置信度统计总数
            if confidence == "高":
                high_confidence_total += 1
            elif confidence == "中":
                medium_confidence_total += 1
            else:
                low_confidence_total += 1
            
            # 按综合得分范围统计
            for range_name, range_data in score_ranges.items():
                min_score, max_score = range_data['range']
                if min_score <= total_score < max_score:
                    range_data['total'] += 1
                    if pred_direction == actual_direction:
                        range_data['correct'] += 1
                    break
            
            predictions.append(pred_change_ratio)
            actual_changes.append(actual_change_ratio)
                
        except Exception as e:
            continue
    
    # 计算统计信息
    if total_predictions > 0:
        predictions = np.array(predictions)
        actual_changes = np.array(actual_changes)
        
        print(f"\n=== 混合预测器统计 ===")
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
            actual_dir = "涨" if actual_changes[i] > 0.0005 else "跌" if actual_changes[i] < -0.0005 else "平"
            correct = "✅" if pred_dir == actual_dir else "❌"
            
            print(f"预测{i+1}: {pred_dir} {predictions[i]*100:.3f}% | 实际: {actual_dir} {actual_changes[i]*100:.3f}% {correct}")
        
        # 分析预测偏差
        bias = predictions.mean() - actual_changes.mean()
        print(f"\n预测偏差: {bias*100:.3f}% (正值表示预测偏乐观，负值表示预测偏悲观)")
        
        # 按置信度分析准确率
        print(f"\n=== 按置信度分析 ===")
        if high_confidence_total > 0:
            high_acc = high_confidence_correct / high_confidence_total * 100
            print(f"高置信度准确率: {high_acc:.2f}% ({high_confidence_correct}/{high_confidence_total})")
        
        if medium_confidence_total > 0:
            medium_acc = medium_confidence_correct / medium_confidence_total * 100
            print(f"中置信度准确率: {medium_acc:.2f}% ({medium_confidence_correct}/{medium_confidence_total})")
        
        if low_confidence_total > 0:
            low_acc = low_confidence_correct / low_confidence_total * 100
            print(f"低置信度准确率: {low_acc:.2f}% ({low_confidence_correct}/{low_confidence_total})")
        
        # 按综合得分范围分析
        print(f"\n=== 按综合得分范围分析 ===")
        range_names = {
            'strong_bullish': '强烈看涨',
            'moderate_bullish': '温和看涨',
            'weak_bullish': '轻微看涨',
            'neutral': '中性',
            'weak_bearish': '轻微看跌',
            'moderate_bearish': '温和看跌',
            'strong_bearish': '强烈看跌'
        }
        
        for range_name, range_data in score_ranges.items():
            if range_data['total'] > 0:
                acc = range_data['correct'] / range_data['total'] * 100
                print(f"{range_names[range_name]}: {acc:.2f}% ({range_data['correct']}/{range_data['total']})")
        
        # 计算加权平均准确率
        total_weighted_correct = high_confidence_correct + medium_confidence_correct * 0.5 + low_confidence_correct * 0.2
        total_weighted = high_confidence_total + medium_confidence_total * 0.5 + low_confidence_total * 0.2
        if total_weighted > 0:
            weighted_acc = total_weighted_correct / total_weighted * 100
            print(f"\n加权平均准确率: {weighted_acc:.2f}%")
        
        # 与AI模型比较
        print(f"\n=== 与AI模型比较 ===")
        ai_accuracy = 51.0  # AI模型的准确率
        hybrid_accuracy = direction_accuracy/total_predictions*100
        improvement = hybrid_accuracy - ai_accuracy
        
        print(f"AI模型准确率: {ai_accuracy:.2f}%")
        print(f"混合预测器准确率: {hybrid_accuracy:.2f}%")
        if improvement > 0:
            print(f"混合预测器提升: +{improvement:.2f}个百分点")
        else:
            print(f"混合预测器差距: {improvement:.2f}个百分点")
        
    else:
        print("没有有效的预测结果")

if __name__ == "__main__":
    evaluate_hybrid_predictor() 