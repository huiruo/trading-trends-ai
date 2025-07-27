#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 btc_1h.json 文件中 _id 字段的连续性
检查是否有缺失的 _id (时间戳)
"""

# python script_analyze_id_continuity.py
import json

def analyze_id_continuity():
    """分析JSON文件中_id字段的连续性"""
    
    # 直接指定文件路径
    file_path = "dataset/btc_1h.json"
    
    print(f"正在分析文件: {file_path}")
    print("=" * 50)
    
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件读取成功，总记录数: {len(data)}")
        
        # 提取所有_id (时间戳)
        ids = []
        for i, record in enumerate(data):
            if '_id' in record:
                ids.append(record['_id'])
            else:
                print(f"警告: 第{i+1}条记录缺少_id字段")
        
        if not ids:
            print("错误: 文件中没有找到任何_id字段")
            return
        
        print(f"找到 {len(ids)} 个_id字段")
        
        # 检查_id类型并转换为整数
        try:
            ids = [int(id_val) for id_val in ids]
            print("✅ ID类型: 整数时间戳")
        except (ValueError, TypeError):
            print("❌ 错误: ID不是有效的整数时间戳")
            return
        
        # 检查是否有重复
        ids_set = set(ids)
        if len(ids_set) != len(ids):
            duplicates = len(ids) - len(ids_set)
            print(f"⚠️  发现 {duplicates} 个重复的时间戳")
        else:
            print("✅ 没有重复的时间戳")
        
        # 检查时间连续性 (假设数据已经按时间排序)
        print("\n正在检查时间连续性...")
        missing_periods = []
        irregular_periods = []
        
        # 一小时 = 3600000 毫秒
        expected_interval = 3600000
        
        for i in range(1, len(ids)):
            current_timestamp = ids[i]
            previous_timestamp = ids[i-1]
            actual_interval = current_timestamp - previous_timestamp
            
            if actual_interval != expected_interval:
                if actual_interval > expected_interval:
                    # 缺失了时间段
                    missing_hours = actual_interval // expected_interval - 1
                    missing_periods.append({
                        'position': i,
                        'previous': previous_timestamp,
                        'current': current_timestamp,
                        'missing_hours': missing_hours,
                        'gap': actual_interval
                    })
                else:
                    # 时间间隔不规则
                    irregular_periods.append({
                        'position': i,
                        'previous': previous_timestamp,
                        'current': current_timestamp,
                        'interval': actual_interval
                    })
        
        # 报告结果
        print(f"\n=== 时间连续性检查结果 ===")
        
        if missing_periods:
            print(f"❌ 发现 {len(missing_periods)} 个缺失时间段:")
            for i, period in enumerate(missing_periods[:5]):  # 只显示前5个
                print(f"  位置{i+1}: 缺失 {period['missing_hours']} 小时")
                print(f"    从: {period['previous']} 到: {period['current']}")
                print(f"    间隔: {period['gap']} 毫秒")
            if len(missing_periods) > 5:
                print(f"  ... 还有 {len(missing_periods) - 5} 个缺失时间段")
        else:
            print("✅ 没有发现缺失的时间段")
        
        if irregular_periods:
            print(f"\n⚠️  发现 {len(irregular_periods)} 个不规则的时间间隔:")
            for i, period in enumerate(irregular_periods[:5]):  # 只显示前5个
                print(f"  位置{i+1}: 间隔 {period['interval']} 毫秒 (应为 {expected_interval} 毫秒)")
                print(f"    从: {period['previous']} 到: {period['current']}")
            if len(irregular_periods) > 5:
                print(f"  ... 还有 {len(irregular_periods) - 5} 个不规则间隔")
        else:
            print("✅ 所有时间间隔都是规则的1小时")
        
        # 统计信息
        print(f"\n=== 统计信息 ===")
        print(f"总记录数: {len(data)}")
        print(f"时间戳数量: {len(ids)}")
        print(f"唯一时间戳: {len(ids_set)}")
        print(f"重复时间戳: {len(ids) - len(ids_set)}")
        print(f"缺失时间段: {len(missing_periods)}")
        print(f"不规则间隔: {len(irregular_periods)}")
        
        # 时间范围信息
        start_time = min(ids)
        end_time = max(ids)
        total_duration = end_time - start_time
        expected_records = total_duration // expected_interval + 1
        
        print(f"\n=== 时间范围信息 ===")
        print(f"开始时间: {start_time}")
        print(f"结束时间: {end_time}")
        print(f"总时长: {total_duration} 毫秒 ({total_duration/3600000:.1f} 小时)")
        print(f"期望记录数: {expected_records}")
        print(f"实际记录数: {len(ids)}")
        print(f"记录数差异: {len(ids) - expected_records}")
        
        # 数据完整性评估
        if len(missing_periods) == 0 and len(irregular_periods) == 0:
            print("\n🎉 数据完整性: 完美！所有时间点都连续且规则")
        elif len(missing_periods) > 0:
            print(f"\n⚠️  数据完整性: 有 {len(missing_periods)} 个时间段缺失")
        else:
            print(f"\n⚠️  数据完整性: 有 {len(irregular_periods)} 个不规则间隔")
        
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式错误 - {e}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    analyze_id_continuity() 