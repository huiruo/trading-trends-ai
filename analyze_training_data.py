# analyze_training_data.py
import pandas as pd
import numpy as np
from preprocess import load_and_preprocess
from technical_indicators import add_technical_indicators
from config_improved import WINDOW_SIZE

def analyze_training_data():
    """分析训练数据的最后一段，检查是否有极端变化"""
    
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
    
    # 添加技术指标
    df = add_technical_indicators(df)
    
    print("=== 训练数据分析 ===")
    print(f"总数据条数: {len(df)}")
    print(f"训练窗口大小: {WINDOW_SIZE}")
    print(f"实际训练样本数: {len(df) - WINDOW_SIZE}")
    
    # 分析最后一段数据（用于预测的数据）
    last_window = df.iloc[-WINDOW_SIZE:]
    print(f"\n=== 最后{WINDOW_SIZE}根K线分析 ===")
    print(f"时间范围: {last_window.iloc[0]['timestamp']} 到 {last_window.iloc[-1]['timestamp']}")
    
    # 计算价格变化
    price_changes = []
    for i in range(1, len(last_window)):
        change = (last_window.iloc[i]['close'] - last_window.iloc[i-1]['close']) / last_window.iloc[i-1]['close'] * 100
        price_changes.append(change)
    
    print(f"\n价格变化统计:")
    print(f"最大涨幅: {max(price_changes):.2f}%")
    print(f"最大跌幅: {min(price_changes):.2f}%")
    print(f"平均变化: {np.mean(price_changes):.2f}%")
    print(f"标准差: {np.std(price_changes):.2f}%")
    
    # 找出极端变化
    extreme_changes = [c for c in price_changes if abs(c) > 3]
    if extreme_changes:
        print(f"\n⚠️ 发现极端变化 (>3%):")
        for i, change in enumerate(extreme_changes):
            print(f"  第{i+1}次: {change:.2f}%")
    else:
        print(f"\n✅ 没有发现极端变化")
    
    # 分析最后几根K线的特征
    print(f"\n=== 最后5根K线详细分析 ===")
    last_5 = df.iloc[-5:]
    for i, row in last_5.iterrows():
        print(f"K线 {i}: 时间={row['timestamp']}, 开={row['open']:.2f}, 高={row['high']:.2f}, 低={row['low']:.2f}, 收={row['close']:.2f}, RSI={row['rsi_14']:.2f}, BB位置={row['bb_position']:.3f}")
    
    # 检查是否有异常的技术指标值
    print(f"\n=== 技术指标异常值检查 ===")
    for col in ['rsi_14', 'bb_position', 'close_open_ratio']:
        if col in df.columns:
            values = df[col].dropna()
            print(f"{col}: 范围=[{values.min():.3f}, {values.max():.3f}], 均值={values.mean():.3f}, 标准差={values.std():.3f}")
            
            # 检查最后一段的指标值
            last_values = last_window[col].dropna()
            print(f"  最后{len(last_values)}个值: 范围=[{last_values.min():.3f}, {last_values.max():.3f}]")

if __name__ == "__main__":
    analyze_training_data() 