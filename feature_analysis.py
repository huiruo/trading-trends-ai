# feature_analysis.py - 特征工程分析脚本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from technical_indicators import add_technical_indicators, get_feature_importance_analysis
from preprocess import load_and_preprocess
from config_improved import *

def analyze_features():
    """分析特征工程效果"""
    print("=== 特征工程分析 ===")
    
    # 加载数据
    df = load_and_preprocess(DATA_PATH)
    df = add_technical_indicators(df)
    
    print(f"原始数据形状: {df.shape}")
    print(f"特征数量: {len(FEATURE_COLUMNS)}")
    
    # 基础统计分析
    print("\n=== 基础统计分析 ===")
    feature_stats = df[FEATURE_COLUMNS].describe()
    print(feature_stats)
    
    # 检查缺失值
    print("\n=== 缺失值检查 ===")
    missing_values = df[FEATURE_COLUMNS].isnull().sum()
    print(missing_values)
    
    # 检查无穷大值
    print("\n=== 无穷大值检查 ===")
    inf_values = np.isinf(df[FEATURE_COLUMNS]).sum()
    print(inf_values)
    
    # 特征相关性分析
    print("\n=== 特征相关性分析 ===")
    correlation_analysis(df)
    
    # 特征重要性分析
    print("\n=== 特征重要性分析 ===")
    importance_analysis(df)
    
    # 特征稳定性分析
    print("\n=== 特征稳定性分析 ===")
    stability_analysis(df)
    
    # 生成可视化报告
    print("\n=== 生成可视化报告 ===")
    generate_visualizations(df)

def correlation_analysis(df: pd.DataFrame):
    """特征相关性分析"""
    feature_df = df[FEATURE_COLUMNS].copy()
    
    # 计算相关性矩阵
    corr_matrix = feature_df.corr()
    
    # 找出高相关性的特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    print(f"发现 {len(high_corr_pairs)} 对高相关性特征 (|r| > 0.8):")
    for pair in high_corr_pairs:
        print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
    
    # 计算每个特征的平均相关性
    avg_corr = corr_matrix.abs().mean().sort_values(ascending=False)
    print(f"\n特征平均相关性 (降序):")
    for feature, corr in avg_corr.items():
        print(f"  {feature}: {corr:.3f}")

def importance_analysis(df: pd.DataFrame):
    """特征重要性分析"""
    feature_df = df[FEATURE_COLUMNS].copy()
    
    # 创建目标变量
    if USE_CLASSIFICATION:
        # 分类目标
        target = create_classification_target(df)
        mi_scores = mutual_info_classif(feature_df, target, random_state=42)
    else:
        # 回归目标
        target = create_regression_target(df)
        mi_scores = mutual_info_regression(feature_df, target, random_state=42)
    
    # 计算特征重要性
    feature_importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)
    
    print("特征重要性 (互信息):")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 识别低重要性特征
    low_importance_threshold = 0.01
    low_importance_features = feature_importance[feature_importance['importance'] < low_importance_threshold]
    
    if len(low_importance_features) > 0:
        print(f"\n⚠️ 发现 {len(low_importance_features)} 个低重要性特征 (< {low_importance_threshold}):")
        for _, row in low_importance_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

def stability_analysis(df: pd.DataFrame):
    """特征稳定性分析"""
    feature_df = df[FEATURE_COLUMNS].copy()
    
    # 计算每个特征的方差
    feature_variance = feature_df.var().sort_values(ascending=False)
    
    print("特征方差 (降序):")
    for feature, variance in feature_variance.items():
        print(f"  {feature}: {variance:.6f}")
    
    # 识别低方差特征
    low_variance_threshold = feature_variance.quantile(0.1)  # 最低10%分位数
    low_variance_features = feature_variance[feature_variance < low_variance_threshold]
    
    if len(low_variance_features) > 0:
        print(f"\n⚠️ 发现 {len(low_variance_features)} 个低方差特征 (< {low_variance_threshold:.6f}):")
        for feature, variance in low_variance_features.items():
            print(f"  {feature}: {variance:.6f}")
    
    # 计算特征的时间稳定性
    print(f"\n=== 特征时间稳定性分析 ===")
    stability_scores = calculate_temporal_stability(df)
    
    print("特征时间稳定性 (越高越稳定):")
    for feature, stability in stability_scores.items():
        print(f"  {feature}: {stability:.3f}")

def calculate_temporal_stability(df: pd.DataFrame) -> dict:
    """计算特征的时间稳定性"""
    stability_scores = {}
    
    for feature in FEATURE_COLUMNS:
        if feature in df.columns:
            # 计算滚动窗口内的标准差
            rolling_std = df[feature].rolling(window=50).std()
            # 稳定性 = 1 / (1 + 平均滚动标准差)
            avg_rolling_std = rolling_std.mean()
            stability = 1 / (1 + avg_rolling_std) if avg_rolling_std > 0 else 1
            stability_scores[feature] = stability
    
    return dict(sorted(stability_scores.items(), key=lambda x: x[1], reverse=True))

def create_classification_target(df: pd.DataFrame) -> np.ndarray:
    """创建分类目标变量"""
    targets = []
    threshold = CLASSIFICATION_THRESHOLD
    
    for i in range(WINDOW_SIZE, len(df)):
        current_close = df.iloc[i-1]['close']
        next_close = df.iloc[i]['close']
        change_ratio = (next_close - current_close) / current_close
        
        if change_ratio < -threshold:
            target = 0  # 跌
        elif change_ratio > threshold:
            target = 2  # 涨
        else:
            target = 1  # 平
        
        targets.append(target)
    
    return np.array(targets)

def create_regression_target(df: pd.DataFrame) -> np.ndarray:
    """创建回归目标变量"""
    targets = []
    
    for i in range(WINDOW_SIZE, len(df)):
        current_close = df.iloc[i-1]['close']
        next_close = df.iloc[i]['close']
        change_ratio = (next_close - current_close) / current_close
        
        # 限制变化率范围
        change_ratio = np.clip(change_ratio, -MAX_CHANGE_RATIO, MAX_CHANGE_RATIO)
        
        targets.append(change_ratio)
    
    return np.array(targets)

def generate_visualizations(df: pd.DataFrame):
    """生成可视化报告"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('特征工程分析报告', fontsize=16)
        
        # 1. 相关性热力图
        feature_df = df[FEATURE_COLUMNS].copy()
        corr_matrix = feature_df.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0,0])
        axes[0,0].set_title('特征相关性热力图')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].tick_params(axis='y', rotation=0)
        
        # 2. 特征重要性条形图
        if USE_CLASSIFICATION:
            target = create_classification_target(df)
            mi_scores = mutual_info_classif(feature_df, target, random_state=42)
        else:
            target = create_regression_target(df)
            mi_scores = mutual_info_regression(feature_df, target, random_state=42)
        
        feature_importance = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': mi_scores
        }).sort_values('importance', ascending=True)
        
        axes[0,1].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[0,1].set_yticks(range(len(feature_importance)))
        axes[0,1].set_yticklabels(feature_importance['feature'])
        axes[0,1].set_title('特征重要性 (互信息)')
        axes[0,1].set_xlabel('重要性得分')
        
        # 3. 特征方差分布
        feature_variance = feature_df.var().sort_values(ascending=True)
        axes[1,0].barh(range(len(feature_variance)), feature_variance.values)
        axes[1,0].set_yticks(range(len(feature_variance)))
        axes[1,0].set_yticklabels(feature_variance.index)
        axes[1,0].set_title('特征方差分布')
        axes[1,0].set_xlabel('方差')
        
        # 4. 特征时间序列示例 (选择前3个重要特征)
        top_features = feature_importance.tail(3)['feature'].tolist()
        for i, feature in enumerate(top_features):
            if feature in df.columns:
                # 只显示最后1000个数据点
                sample_data = df[feature].tail(1000)
                axes[1,1].plot(sample_data.index, sample_data.values, 
                              label=feature, alpha=0.7)
        
        axes[1,1].set_title('重要特征时间序列示例')
        axes[1,1].set_xlabel('时间索引')
        axes[1,1].set_ylabel('特征值')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_analysis_report.png', dpi=300, bbox_inches='tight')
        print("✅ 可视化报告已保存为: feature_analysis_report.png")
        
    except Exception as e:
        print(f"⚠️ 生成可视化报告失败: {e}")

def recommend_feature_improvements():
    """推荐特征改进建议"""
    print("\n=== 特征改进建议 ===")
    
    # 加载数据进行分析
    df = load_and_preprocess(DATA_PATH)
    df = add_technical_indicators(df)
    
    # 分析当前特征
    feature_analysis = get_feature_importance_analysis(df)
    
    print("📊 当前特征分析:")
    print(f"  总特征数: {feature_analysis['total_features']}")
    print(f"  高相关性特征对: {len(feature_analysis['high_correlation_pairs'])}")
    
    # 建议
    print("\n💡 改进建议:")
    
    if len(feature_analysis['high_correlation_pairs']) > 0:
        print("  1. 移除高相关性特征对中的冗余特征")
        for pair in feature_analysis['high_correlation_pairs']:
            print(f"     - 考虑移除 {pair['feature1']} 或 {pair['feature2']} (相关性: {pair['correlation']:.3f})")
    
    # 检查低方差特征
    low_variance_features = []
    for feature, variance in feature_analysis['feature_variance'].items():
        if variance < 0.001:  # 低方差阈值
            low_variance_features.append(feature)
    
    if low_variance_features:
        print(f"  2. 考虑移除低方差特征: {low_variance_features}")
    
    print("  3. 考虑添加以下新特征:")
    print("     - 价格动量特征 (不同时间窗口)")
    print("     - 成交量相关特征")
    print("     - 波动率特征")
    print("     - 市场情绪指标")
    
    print("  4. 特征工程优化:")
    print("     - 使用对数收益率替代原始价格")
    print("     - 应用z-score标准化")
    print("     - 考虑特征选择算法")

if __name__ == "__main__":
    analyze_features()
    recommend_feature_improvements() 