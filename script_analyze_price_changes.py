# analyze_price_changes.py
import pandas as pd
import numpy as np
from config_improved import CLASSIFICATION_THRESHOLD

# python script_analyze_price_changes.py
def analyze_price_changes():
    """分析价格变化的分布"""
    
    # 加载数据
    from config_improved import DATA_PATH
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
    
    print("=== 价格变化分析 ===")
    print(f"总数据条数: {len(df)}")
    
    # 计算价格变化
    df['price_change'] = df['close'].pct_change()
    df['price_change_pct'] = df['price_change'] * 100
    
    # 移除第一行的NaN
    df = df.dropna()
    
    print(f"有效价格变化数据: {len(df)}")
    
    # 基本统计
    print(f"\n价格变化统计:")
    print(f"平均变化: {df['price_change_pct'].mean():.4f}%")
    print(f"中位数变化: {df['price_change_pct'].median():.4f}%")
    print(f"标准差: {df['price_change_pct'].std():.4f}%")
    print(f"最大涨幅: {df['price_change_pct'].max():.4f}%")
    print(f"最大跌幅: {df['price_change_pct'].min():.4f}%")
    
    # 分类分析
    print(f"\n=== 分类分析 (阈值: {CLASSIFICATION_THRESHOLD*100:.1f}%) ===")
    
    # 按分类阈值分类
    df['direction'] = '平'
    df.loc[df['price_change'] < -CLASSIFICATION_THRESHOLD, 'direction'] = '跌'
    df.loc[df['price_change'] > CLASSIFICATION_THRESHOLD, 'direction'] = '涨'
    
    direction_counts = df['direction'].value_counts()
    direction_pcts = df['direction'].value_counts(normalize=True) * 100
    
    print("方向分布:")
    for direction in ['跌', '平', '涨']:
        count = direction_counts.get(direction, 0)
        pct = direction_pcts.get(direction, 0)
        print(f"  {direction}: {count}次 ({pct:.2f}%)")
    
    # 分析不同阈值下的分布
    print(f"\n=== 不同阈值下的分布分析 ===")
    thresholds = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    
    for threshold in thresholds:
        temp_df = df.copy()
        temp_df['temp_direction'] = '平'
        temp_df.loc[temp_df['price_change'] < -threshold, 'temp_direction'] = '跌'
        temp_df.loc[temp_df['price_change'] > threshold, 'temp_direction'] = '涨'
        
        counts = temp_df['temp_direction'].value_counts()
        pcts = temp_df['temp_direction'].value_counts(normalize=True) * 100
        
        print(f"阈值 {threshold*100:.1f}%:")
        for direction in ['跌', '平', '涨']:
            count = counts.get(direction, 0)
            pct = pcts.get(direction, 0)
            print(f"  {direction}: {count}次 ({pct:.1f}%)")
        print()
    
    # 最近100根K线的分析
    print(f"\n=== 最近100根K线分析 ===")
    recent_100 = df.tail(100)
    recent_direction_counts = recent_100['direction'].value_counts()
    recent_direction_pcts = recent_100['direction'].value_counts(normalize=True) * 100
    
    print("最近100根K线方向分布:")
    for direction in ['跌', '平', '涨']:
        count = recent_direction_counts.get(direction, 0)
        pct = recent_direction_pcts.get(direction, 0)
        print(f"  {direction}: {count}次 ({pct:.1f}%)")
    
    print(f"\n最近100根K线平均变化: {recent_100['price_change_pct'].mean():.4f}%")
    print(f"最近100根K线最大变化: {recent_100['price_change_pct'].max():.4f}%")
    print(f"最近100根K线最小变化: {recent_100['price_change_pct'].min():.4f}%")

if __name__ == "__main__":
    analyze_price_changes() 