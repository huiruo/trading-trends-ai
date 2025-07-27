# analyze_price_changes.py - 分析价格变化分布
import pandas as pd
import numpy as np
from config_improved import DATA_PATH

def analyze_price_changes():
    print("=== 价格变化分析 ===")
    
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
    
    # 计算价格变化
    df['price_change'] = df['close'].pct_change() * 100
    
    print(f"总数据条数: {len(df)}")
    print(f"有效价格变化数据: {len(df.dropna())}")
    
    # 基本统计
    changes = df['price_change'].dropna()
    print(f"\n价格变化统计:")
    print(f"平均变化: {changes.mean():.4f}%")
    print(f"中位数变化: {changes.median():.4f}%")
    print(f"标准差: {changes.std():.4f}%")
    print(f"最大涨幅: {changes.max():.4f}%")
    print(f"最大跌幅: {changes.min():.4f}%")
    
    # 分位数分析
    print(f"\n分位数分析:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(changes, p)
        print(f"  {p}%分位数: {value:.4f}%")
    
    # 极端变化分析
    print(f"\n极端变化分析:")
    extreme_up = changes[changes > 3]
    extreme_down = changes[changes < -3]
    print(f"涨幅 > 3%: {len(extreme_up)}次 ({len(extreme_up)/len(changes)*100:.2f}%)")
    print(f"跌幅 < -3%: {len(extreme_down)}次 ({len(extreme_down)/len(changes)*100:.2f}%)")
    
    if len(extreme_up) > 0:
        print(f"最大涨幅: {extreme_up.max():.4f}%")
    if len(extreme_down) > 0:
        print(f"最大跌幅: {extreme_down.min():.4f}%")
    
    # 最近100根K线的变化
    print(f"\n最近100根K线的变化:")
    recent_changes = changes.tail(100)
    print(f"平均变化: {recent_changes.mean():.4f}%")
    print(f"最大变化: {recent_changes.max():.4f}%")
    print(f"最小变化: {recent_changes.min():.4f}%")

if __name__ == "__main__":
    analyze_price_changes() 