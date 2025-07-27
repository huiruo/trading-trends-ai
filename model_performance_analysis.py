# model_performance_analysis.py - 模型性能分析脚本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import load_and_preprocess, create_sequences
from technical_indicators import add_technical_indicators
from model.model import LSTMModel
from config_improved import *
import torch

def analyze_model_performance():
    """分析模型性能并提供改进建议"""
    print("=== 模型性能深度分析 ===")
    
    # 加载数据
    df = load_and_preprocess(DATA_PATH)
    df = add_technical_indicators(df)
    
    # 创建序列
    X, _ = create_sequences(df, window_size=WINDOW_SIZE)
    
    if USE_CLASSIFICATION:
        analyze_classification_performance(X, df)
    else:
        analyze_regression_performance(X, df)

def analyze_classification_performance(X: np.ndarray, df: pd.DataFrame):
    """分析分类模型性能"""
    print("\n=== 分类模型性能分析 ===")
    
    # 加载模型
    model = LSTMModel(
        input_size=X.shape[2], 
        hidden_size=64, 
        num_layers=2, 
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    # 预测
    predictions = []
    actual_changes = []
    confidences = []
    class_predictions = []
    class_actuals = []
    
    with torch.no_grad():
        for i in range(len(X)):
            x_tensor = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)
            pred_probs = model(x_tensor)
            pred_class = torch.argmax(pred_probs, dim=1).item()
            confidence = torch.max(pred_probs, dim=1).values.item()
            
            # 实际变化
            current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
            actual_close = df.iloc[i + WINDOW_SIZE]['close']
            actual_change_ratio = (actual_close - current_close) / current_close
            
            # 实际分类
            if actual_change_ratio < -CLASSIFICATION_THRESHOLD:
                actual_class = 0  # 跌
            elif actual_change_ratio > CLASSIFICATION_THRESHOLD:
                actual_class = 2  # 涨
            else:
                actual_class = 1  # 平
            
            predictions.append(pred_class)
            actual_changes.append(actual_change_ratio)
            confidences.append(confidence)
            class_predictions.append(pred_class)
            class_actuals.append(actual_class)
    
    # 转换为numpy数组
    predictions = np.array(predictions)
    actual_changes = np.array(actual_changes)
    confidences = np.array(confidences)
    class_predictions = np.array(class_predictions)
    class_actuals = np.array(class_actuals)
    
    # 1. 分类分布分析
    print("\n1. 分类分布分析:")
    pred_dist = np.bincount(predictions, minlength=3)
    actual_dist = np.bincount(class_actuals, minlength=3)
    
    labels = ['跌', '平', '涨']
    for i, label in enumerate(labels):
        print(f"  {label}: 预测={pred_dist[i]} ({pred_dist[i]/len(predictions)*100:.1f}%), "
              f"实际={actual_dist[i]} ({actual_dist[i]/len(class_actuals)*100:.1f}%)")
    
    # 2. 置信度分析
    print("\n2. 置信度分析:")
    print(f"  平均置信度: {confidences.mean():.3f}")
    print(f"  置信度标准差: {confidences.std():.3f}")
    print(f"  最高置信度: {confidences.max():.3f}")
    print(f"  最低置信度: {confidences.min():.3f}")
    
    # 按置信度分组分析准确率
    confidence_bins = [0.3, 0.5, 0.7, 0.9, 1.0]
    for i in range(len(confidence_bins)-1):
        mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
        if mask.sum() > 0:
            accuracy = (predictions[mask] == class_actuals[mask]).mean()
            print(f"  置信度 {confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}: "
                  f"准确率={accuracy:.3f} (样本数={mask.sum()})")
    
    # 3. 错误模式分析
    print("\n3. 错误模式分析:")
    errors = predictions != class_actuals
    error_indices = np.where(errors)[0]
    
    if len(error_indices) > 0:
        print(f"  总错误数: {len(error_indices)}")
        
        # 分析错误类型
        error_types = []
        for idx in error_indices:
            actual = class_actuals[idx]
            pred = predictions[idx]
            if actual == 0 and pred == 2:  # 实际跌，预测涨
                error_types.append('跌->涨')
            elif actual == 2 and pred == 0:  # 实际涨，预测跌
                error_types.append('涨->跌')
            elif actual == 1:  # 实际平
                if pred == 0:
                    error_types.append('平->跌')
                else:
                    error_types.append('平->涨')
            elif pred == 1:  # 预测平
                if actual == 0:
                    error_types.append('跌->平')
                else:
                    error_types.append('涨->平')
        
        from collections import Counter
        error_counts = Counter(error_types)
        print("  错误类型分布:")
        for error_type, count in error_counts.most_common():
            print(f"    {error_type}: {count} ({count/len(error_indices)*100:.1f}%)")
    
    # 4. 时间序列分析
    print("\n4. 时间序列分析:")
    # 分析最近100个预测的准确率
    recent_accuracy = (predictions[-100:] == class_actuals[-100:]).mean()
    print(f"  最近100个预测准确率: {recent_accuracy:.3f}")
    
    # 分析不同时间段的准确率
    segments = 5
    segment_size = len(predictions) // segments
    for i in range(segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < segments - 1 else len(predictions)
        segment_accuracy = (predictions[start_idx:end_idx] == class_actuals[start_idx:end_idx]).mean()
        print(f"  时间段 {i+1}: 准确率={segment_accuracy:.3f}")
    
    # 5. 生成改进建议
    print("\n5. 改进建议:")
    
    # 检查是否过度预测某个类别
    pred_ratios = pred_dist / len(predictions)
    actual_ratios = actual_dist / len(class_actuals)
    
    for i, label in enumerate(labels):
        ratio_diff = pred_ratios[i] - actual_ratios[i]
        if abs(ratio_diff) > 0.1:  # 差异超过10%
            if ratio_diff > 0:
                print(f"  ⚠️ 过度预测{label} (+{ratio_diff*100:.1f}%)")
            else:
                print(f"  ⚠️ 预测不足{label} ({ratio_diff*100:.1f}%)")
    
    # 检查置信度分布
    if confidences.mean() < 0.5:
        print("  ⚠️ 平均置信度过低，模型不够确定")
    
    if confidences.std() < 0.1:
        print("  ⚠️ 置信度变化太小，模型可能过于保守")
    
    # 建议调整阈值
    if pred_dist[1] / len(predictions) > 0.6:  # 平盘预测超过60%
        print("  💡 建议降低分类阈值，减少平盘预测")
    
    # 建议增加训练数据
    if len(predictions) < 5000:
        print("  💡 建议增加训练数据量")
    
    # 6. 生成可视化
    generate_performance_visualizations(predictions, class_actuals, confidences, actual_changes)

def analyze_regression_performance(X: np.ndarray, df: pd.DataFrame):
    """分析回归模型性能"""
    print("\n=== 回归模型性能分析 ===")
    
    # 加载模型
    model = LSTMModel(
        input_size=X.shape[2], 
        hidden_size=64, 
        num_layers=2, 
        num_classes=1
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    # 预测
    predictions = []
    actual_changes = []
    
    with torch.no_grad():
        for i in range(len(X)):
            x_tensor = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)
            pred_normalized = model(x_tensor).item()
            
            # 转换预测值
            pred_change_ratio = (pred_normalized * 2 * MAX_CHANGE_RATIO) - MAX_CHANGE_RATIO
            
            # 实际变化
            current_close = df.iloc[i + WINDOW_SIZE - 1]['close']
            actual_close = df.iloc[i + WINDOW_SIZE]['close']
            actual_change_ratio = (actual_close - current_close) / current_close
            
            predictions.append(pred_change_ratio)
            actual_changes.append(actual_change_ratio)
    
    predictions = np.array(predictions)
    actual_changes = np.array(actual_changes)
    
    # 分析预测误差
    errors = predictions - actual_changes
    
    print(f"\n预测误差分析:")
    print(f"  平均误差: {errors.mean()*100:.3f}%")
    print(f"  误差标准差: {errors.std()*100:.3f}%")
    print(f"  平均绝对误差: {np.abs(errors).mean()*100:.3f}%")
    print(f"  最大误差: {errors.max()*100:.3f}%")
    print(f"  最小误差: {errors.min()*100:.3f}%")
    
    # 方向准确率
    pred_directions = predictions > 0
    actual_directions = actual_changes > 0
    direction_accuracy = (pred_directions == actual_directions).mean()
    print(f"\n方向准确率: {direction_accuracy:.3f}")
    
    # 分析不同幅度区间的准确率
    print(f"\n不同幅度区间的方向准确率:")
    magnitude_bins = [0.001, 0.005, 0.01, 0.02, 0.05]
    for i in range(len(magnitude_bins)-1):
        mask = (np.abs(actual_changes) >= magnitude_bins[i]) & (np.abs(actual_changes) < magnitude_bins[i+1])
        if mask.sum() > 0:
            accuracy = (pred_directions[mask] == actual_directions[mask]).mean()
            print(f"  {magnitude_bins[i]*100:.1f}%-{magnitude_bins[i+1]*100:.1f}%: "
                  f"准确率={accuracy:.3f} (样本数={mask.sum()})")

def generate_performance_visualizations(predictions, actuals, confidences, actual_changes):
    """生成性能可视化图表"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能分析报告', fontsize=16)
        
        # 1. 混淆矩阵
        cm = confusion_matrix(actuals, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['跌', '平', '涨'], 
                   yticklabels=['跌', '平', '涨'], ax=axes[0,0])
        axes[0,0].set_title('混淆矩阵')
        axes[0,0].set_xlabel('预测')
        axes[0,0].set_ylabel('实际')
        
        # 2. 置信度分布
        axes[0,1].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('预测置信度分布')
        axes[0,1].set_xlabel('置信度')
        axes[0,1].set_ylabel('频次')
        axes[0,1].axvline(confidences.mean(), color='red', linestyle='--', 
                          label=f'平均: {confidences.mean():.3f}')
        axes[0,1].legend()
        
        # 3. 准确率随时间变化
        window_size = 100
        accuracies = []
        for i in range(0, len(predictions), window_size):
            end_idx = min(i + window_size, len(predictions))
            accuracy = (predictions[i:end_idx] == actuals[i:end_idx]).mean()
            accuracies.append(accuracy)
        
        axes[1,0].plot(accuracies)
        axes[1,0].set_title('准确率随时间变化')
        axes[1,0].set_xlabel('时间窗口')
        axes[1,0].set_ylabel('准确率')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 实际变化分布
        axes[1,1].hist(actual_changes * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('实际价格变化分布')
        axes[1,1].set_xlabel('变化幅度 (%)')
        axes[1,1].set_ylabel('频次')
        axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ 性能分析图表已保存为: model_performance_analysis.png")
        
    except Exception as e:
        print(f"⚠️ 生成可视化失败: {e}")

if __name__ == "__main__":
    analyze_model_performance() 