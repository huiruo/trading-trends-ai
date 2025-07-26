# feature_analysis.py - 特征分析工具
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess, create_sequences
from technical_indicators import add_technical_indicators
from config_improved import FEATURE_COLUMNS, WINDOW_SIZE
import seaborn as sns

def analyze_features(df):
    """分析特征，找出可能导致极端预测的特征"""
    print("=== 特征分析报告 ===\n")
    
    # 添加技术指标
    df_with_indicators = add_technical_indicators(df)
    
    # 计算每根K线的涨跌幅
    df_with_indicators['change_ratio'] = df_with_indicators['close'].pct_change()
    
    print("1. 价格变化统计:")
    print(f"   最大涨幅: {df_with_indicators['change_ratio'].max()*100:.2f}%")
    print(f"   最大跌幅: {df_with_indicators['change_ratio'].min()*100:.2f}%")
    print(f"   平均涨跌幅: {df_with_indicators['change_ratio'].abs().mean()*100:.2f}%")
    print(f"   标准差: {df_with_indicators['change_ratio'].std()*100:.2f}%\n")
    
    # 分析每个特征的统计信息
    print("2. 特征统计信息:")
    feature_stats = {}
    
    for feature in FEATURE_COLUMNS:
        if feature in df_with_indicators.columns:
            values = df_with_indicators[feature].dropna()
            feature_stats[feature] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'range': values.max() - values.min(),
                'has_inf': np.isinf(values).any(),
                'has_nan': values.isna().any()
            }
            
            print(f"   {feature}:")
            print(f"     范围: [{values.min():.4f}, {values.max():.4f}]")
            print(f"     均值: {values.mean():.4f}, 标准差: {values.std():.4f}")
            if np.isinf(values).any():
                print(f"     ⚠️ 包含无穷大值")
            if values.isna().any():
                print(f"     ⚠️ 包含NaN值")
    
    # 找出异常特征
    print("\n3. 异常特征检测:")
    problematic_features = []
    
    for feature, stats in feature_stats.items():
        # 检查是否有无穷大值
        if stats['has_inf']:
            problematic_features.append((feature, "包含无穷大值"))
            print(f"   ❌ {feature}: 包含无穷大值")
        
        # 检查是否有NaN值
        if stats['has_nan']:
            problematic_features.append((feature, "包含NaN值"))
            print(f"   ❌ {feature}: 包含NaN值")
        
        # 检查数值范围是否异常
        if stats['range'] > 1000000:  # 范围过大
            problematic_features.append((feature, "数值范围过大"))
            print(f"   ⚠️ {feature}: 数值范围过大 ({stats['range']:.2f})")
        
        # 检查标准差是否异常
        if stats['std'] > 10000:  # 标准差过大
            problematic_features.append((feature, "标准差过大"))
            print(f"   ⚠️ {feature}: 标准差过大 ({stats['std']:.2f})")
    
    # 特征与价格变化的相关性分析
    print("\n4. 特征与价格变化的相关性:")
    correlations = {}
    
    for feature in FEATURE_COLUMNS:
        if feature in df_with_indicators.columns:
            corr = df_with_indicators[feature].corr(df_with_indicators['change_ratio'])
            correlations[feature] = corr
            print(f"   {feature}: {corr:.4f}")
    
    # 找出相关性最强的特征
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n   相关性最强的特征: {sorted_correlations[0][0]} ({sorted_correlations[0][1]:.4f})")
    print(f"   相关性最弱的特征: {sorted_correlations[-1][0]} ({sorted_correlations[-1][1]:.4f})")
    
    return feature_stats, correlations, problematic_features

def suggest_feature_fixes(problematic_features, correlations):
    """根据分析结果建议特征修复方案"""
    print("\n=== 特征修复建议 ===\n")
    
    if not problematic_features:
        print("✅ 没有发现明显的问题特征")
    else:
        print("建议修复以下特征:")
        for feature, issue in problematic_features:
            print(f"   - {feature}: {issue}")
    
    print("\n建议保留的特征 (按相关性排序):")
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (feature, corr) in enumerate(sorted_correlations[:10]):  # 只显示前10个
        print(f"   {i+1}. {feature}: {corr:.4f}")

def create_feature_visualization(df):
    """创建特征可视化图表"""
    df_with_indicators = add_technical_indicators(df)
    df_with_indicators['change_ratio'] = df_with_indicators['close'].pct_change()
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 价格变化分布
    axes[0, 0].hist(df_with_indicators['change_ratio'].dropna() * 100, bins=50, alpha=0.7)
    axes[0, 0].set_title('价格变化分布')
    axes[0, 0].set_xlabel('涨跌幅 (%)')
    axes[0, 0].set_ylabel('频次')
    
    # 2. 特征相关性热力图
    feature_cols = [col for col in FEATURE_COLUMNS if col in df_with_indicators.columns]
    corr_matrix = df_with_indicators[feature_cols + ['change_ratio']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
    axes[0, 1].set_title('特征相关性热力图')
    
    # 3. 主要特征的时间序列
    if 'rsi_14' in df_with_indicators.columns:
        axes[1, 0].plot(df_with_indicators['rsi_14'])
        axes[1, 0].set_title('RSI时间序列')
        axes[1, 0].set_ylabel('RSI')
    
    # 4. 价格与预测目标
    axes[1, 1].plot(df_with_indicators['close'])
    axes[1, 1].set_title('收盘价时间序列')
    axes[1, 1].set_ylabel('价格')
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 特征分析图表已保存为 'feature_analysis.png'")

if __name__ == "__main__":
    # 加载数据
    df = pd.read_csv("dataset/btc_1h.csv")
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
    
    # 分析特征
    feature_stats, correlations, problematic_features = analyze_features(df)
    
    # 建议修复方案
    suggest_feature_fixes(problematic_features, correlations)
    
    # 创建可视化
    create_feature_visualization(df) 