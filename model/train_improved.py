# model/train_improved.py - 统一训练脚本（分类/回归）
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocess import load_and_preprocess, create_sequences
from technical_indicators import add_technical_indicators, get_feature_importance_analysis
from model.model import LSTMModel
from config_improved import *
import numpy as np
import pandas as pd # Added missing import for pandas

def create_labels_classification(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """为分类模型创建标签"""
    labels = []
    threshold = CLASSIFICATION_THRESHOLD
    
    for i in range(window_size, len(df)):
        current_close = df.iloc[i-1]['close']
        next_close = df.iloc[i]['close']
        change_ratio = (next_close - current_close) / current_close
        
        # 2分类：跌、涨
        if change_ratio < 0:
            label = 0  # 跌
        else:
            label = 1  # 涨
        
        labels.append(label)
    
    return np.array(labels)

def create_labels_regression(df: pd.DataFrame, window_size: int) -> np.ndarray:
    """为回归模型创建标签"""
    labels = []
    
    for i in range(window_size, len(df)):
        current_close = df.iloc[i-1]['close']
        next_close = df.iloc[i]['close']
        change_ratio = (next_close - current_close) / current_close
        
        # 限制变化率范围
        change_ratio = np.clip(change_ratio, -MAX_CHANGE_RATIO, MAX_CHANGE_RATIO)
        
        # 归一化到0-1范围
        normalized_change = (change_ratio + MAX_CHANGE_RATIO) / (2 * MAX_CHANGE_RATIO)
        
        labels.append(normalized_change)
    
    return np.array(labels)

def train_improved(csv_path: str, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    """统一训练函数"""
    
    print("=== 模型训练开始 ===")
    print(f"模型类型: {'分类' if USE_CLASSIFICATION else '回归'}")
    print(f"特征数量: {len(FEATURE_COLUMNS)}")
    print(f"特征列表: {FEATURE_COLUMNS}")
    
    # 加载数据
    df = load_and_preprocess(csv_path)
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    # 检查数据质量
    print("\n=== 数据质量检查 ===")
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"NaN值数量: {nan_count}")
    print(f"无穷大值数量: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("⚠️ 发现NaN或无穷大值，正在清理...")
        df = df.ffill().bfill()
        df = df.replace([np.inf, -np.inf], 0)
        print("数据清理完成")
    
    # 特征重要性分析
    print("\n=== 特征重要性分析 ===")
    feature_analysis = get_feature_importance_analysis(df)
    print(f"总特征数: {feature_analysis['total_features']}")
    
    if feature_analysis['high_correlation_pairs']:
        print("⚠️ 发现高相关性特征对:")
        for pair in feature_analysis['high_correlation_pairs']:
            print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    
    # 创建序列
    X, _ = create_sequences(df, window_size=WINDOW_SIZE)
    
    # 根据模型类型创建标签
    if USE_CLASSIFICATION:
        y = create_labels_classification(df, WINDOW_SIZE)
        num_classes = NUM_CLASSES
        print(f"\n分类标签分布:")
        unique, counts = np.unique(y, return_counts=True)
        for i, count in zip(unique, counts):
            label_name = ['跌', '涨'][i]
            print(f"  {label_name}: {count} ({count/len(y)*100:.1f}%)")
    else:
        y = create_labels_regression(df, WINDOW_SIZE)
        num_classes = 1
        print(f"\n回归标签统计:")
        print(f"  最小值: {y.min():.4f}")
        print(f"  最大值: {y.max():.4f}")
        print(f"  均值: {y.mean():.4f}")
        print(f"  标准差: {y.std():.4f}")
    
    print(f"\n训练样本数: {len(X)}")
    print(f"特征维度: {X.shape[2]}")
    
    # 创建模型
    model = LSTMModel(
        input_size=X.shape[2], 
        hidden_size=64, 
        num_layers=2, 
        num_classes=num_classes
    )
    
    # 删除旧模型
    if os.path.exists(MODEL_PATH):
        print(f"🗑️ 删除旧模型，重新训练")
        os.remove(MODEL_PATH)
    
    # 设置损失函数和优化器
    if USE_CLASSIFICATION:
        loss_fn = nn.CrossEntropyLoss()
        print("使用分类损失函数: CrossEntropyLoss")
    else:
        loss_fn = nn.MSELoss()
        print("使用回归损失函数: MSELoss")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 准备数据
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if USE_CLASSIFICATION:
        y_tensor = torch.tensor(y, dtype=torch.long)  # 分类使用long类型
    else:
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # 回归使用float类型
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    print(f"\n=== 开始训练 ===")
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        model.train()
        
        for xb, yb in dataloader:
            # 检查数据质量
            if torch.isnan(xb).any() or torch.isnan(yb).any():
                print(f"⚠️ Epoch {epoch+1}: 发现NaN值，跳过批次")
                continue
            
            pred = model(xb)
            loss = loss_fn(pred, yb)
            
            if torch.isnan(loss):
                print(f"⚠️ Epoch {epoch+1}: Loss为NaN，跳过批次")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}, Patience: {patience_counter}")
            
            # 早停
            if patience_counter >= max_patience:
                print(f"🛑 早停触发，在epoch {epoch+1}停止训练")
                break
        else:
            print(f"Epoch {epoch+1}/{epochs}: 所有批次都包含NaN，跳过")
    
    print(f"\n✅ 训练完成！模型保存到: {MODEL_PATH}")
    print(f"最终损失: {best_loss:.6f}")

if __name__ == "__main__":
    train_improved(DATA_PATH) 