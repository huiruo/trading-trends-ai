#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 btc_1h.csv 文件按 _id 时间戳进行排序
从旧到新排序
"""

# script_sort_csv_by_timestamp.py
import pandas as pd
import os

def sort_csv_by_timestamp():
    """对CSV文件按时间戳排序"""
    
    # 文件路径
    input_file = "dataset-sorted/btc_1h.csv"
    output_file = "dataset-sorted/btc_1h_sorted.csv"
    
    print(f"正在处理文件: {input_file}")
    print("=" * 50)
    
    try:
        # 读取CSV文件
        print("正在读取CSV文件...")
        df = pd.read_csv(input_file)
        
        print(f"文件读取成功，总记录数: {len(df)}")
        print(f"列名: {list(df.columns)}")
        
        # 检查_id列是否存在
        if '_id' not in df.columns:
            print("❌ 错误: 文件中没有找到 '_id' 列")
            return
        
        # 显示排序前的数据范围
        print(f"\n排序前的时间范围:")
        print(f"最早时间: {df['_id'].min()}")
        print(f"最晚时间: {df['_id'].max()}")
        
        # 检查原始数据是否已经排序
        print(f"\n=== 检查原始数据是否已排序 ===")
        is_original_sorted = df['_id'].is_monotonic_increasing
        if is_original_sorted:
            print("✅ 原始数据已经是按时间戳排序的（从旧到新）")
        else:
            print("❌ 原始数据未排序，需要重新排序")
            
            # 检查是否是完全逆序（从新到旧）
            is_reverse_sorted = df['_id'].is_monotonic_decreasing
            if is_reverse_sorted:
                print("⚠️  原始数据是逆序的（从新到旧），需要反转")
            else:
                print("⚠️  原始数据是乱序的，需要重新排序")
        
        # 检查是否有重复的时间戳
        duplicates = df['_id'].duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  发现 {duplicates} 个重复的时间戳")
        else:
            print("✅ 没有重复的时间戳")
        
        # 按_id排序（从旧到新）
        print("\n正在按时间戳排序（从旧到新）...")
        df_sorted = df.sort_values('_id', ascending=True)
        
        # 显示排序后的数据范围
        print(f"\n排序后的时间范围:")
        print(f"最早时间: {df_sorted['_id'].min()}")
        print(f"最晚时间: {df_sorted['_id'].max()}")
        
        # 检查排序是否正确
        is_sorted = df_sorted['_id'].is_monotonic_increasing
        if is_sorted:
            print("✅ 排序验证成功：时间戳已正确排序")
        else:
            print("❌ 排序验证失败：时间戳排序有问题")
        
        # 检查排序前后是否有变化
        if is_original_sorted:
            print("✅ 原始数据已经排序，无需重新排序")
            # 如果已经排序，直接复制原文件
            import shutil
            shutil.copy2(input_file, output_file)
            print(f"文件已复制到: {output_file}")
        else:
            # 保存排序后的文件
            print(f"\n正在保存排序后的文件: {output_file}")
            df_sorted.to_csv(output_file, index=False)
        
        # 验证文件大小
        original_size = os.path.getsize(input_file)
        sorted_size = os.path.getsize(output_file)
        
        print(f"\n=== 文件信息 ===")
        print(f"原始文件大小: {original_size:,} 字节")
        print(f"排序文件大小: {sorted_size:,} 字节")
        print(f"记录数: {len(df_sorted)}")
        
        # 显示前几行和后几行数据
        print(f"\n=== 排序后的数据预览 ===")
        print("前5行数据:")
        print(df_sorted.head().to_string(index=False))
        
        print(f"\n后5行数据:")
        print(df_sorted.tail().to_string(index=False))
        
        # 检查时间间隔
        print(f"\n=== 时间间隔检查 ===")
        time_diffs = df_sorted['_id'].diff().dropna()
        expected_interval = 3600000  # 1小时 = 3600000毫秒
        
        irregular_intervals = time_diffs[time_diffs != expected_interval]
        if len(irregular_intervals) > 0:
            print(f"⚠️  发现 {len(irregular_intervals)} 个不规则的时间间隔")
            print("前5个不规则间隔:")
            for i, (idx, diff) in enumerate(irregular_intervals.head().items()):
                print(f"  位置 {idx}: 间隔 {diff} 毫秒 (应为 {expected_interval} 毫秒)")
        else:
            print("✅ 所有时间间隔都是规则的1小时")
        
        # 总结
        print(f"\n=== 总结 ===")
        if is_original_sorted:
            print("🎉 原始数据已经正确排序，无需处理")
        else:
            print("🎉 排序完成！文件已保存为: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ 错误: 文件 {input_file} 不存在")
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    sort_csv_by_timestamp() 