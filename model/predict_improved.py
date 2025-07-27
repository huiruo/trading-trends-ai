# model/predict_improved.py - 统一预测脚本（分类/回归）
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import numpy as np
from model.model import LSTMModel
from preprocess import load_and_preprocess, create_sequences, load_scaler
from technical_indicators import add_technical_indicators
from config_improved import *

def predict_next_candle_improved(df: pd.DataFrame):
    """统一的预测函数"""
    
    print("=== 模型预测开始 ===")
    print(f"模型类型: {'分类' if USE_CLASSIFICATION else '回归'}")
    
    # 加载已保存的scaler
    loaded_scaler = load_scaler()
    if loaded_scaler is None:
        print("⚠️ No scaler found. 请先重新训练模型 (python -m model.train_improved)")
        return None
    
    # 检查scaler特征名与当前特征名是否一致
    scaler_features = getattr(loaded_scaler, 'feature_names_in_', None)
    if scaler_features is not None:
        if list(scaler_features) != list(FEATURE_COLUMNS):
            print("❌ 检测到 scaler.pkl 特征与当前 FEATURE_COLUMNS 不一致！")
            print(f"scaler特征: {list(scaler_features)}")
            print(f"当前特征: {list(FEATURE_COLUMNS)}")
            print("请删除 model/scaler.pkl 并重新训练模型: python -m model.train_improved")
            return None
    
    # 添加技术指标
    df_with_indicators = add_technical_indicators(df)
    
    # 使用已保存的scaler进行标准化
    df_processed = df_with_indicators.copy()
    df_processed[FEATURE_COLUMNS] = loaded_scaler.transform(df_with_indicators[FEATURE_COLUMNS])

    X, _ = create_sequences(df_processed, window_size=WINDOW_SIZE)
    
    if len(X) == 0:
        print("⚠️ Not enough data for prediction. Need at least {} data points.".format(WINDOW_SIZE + 1))
        return None

    X_latest = X[-1]
    X_tensor = torch.tensor(X_latest, dtype=torch.float32).unsqueeze(0)

    # 根据配置加载对应类型的模型
    if USE_CLASSIFICATION:
        return predict_classification(X_tensor, df)
    else:
        return predict_regression(X_tensor, df)

def predict_classification(X_tensor: torch.Tensor, df: pd.DataFrame):
    """分类模型预测"""
    try:
        # 加载分类模型
        model = LSTMModel(
            input_size=X_tensor.shape[2], 
            hidden_size=64, 
            num_layers=2, 
            num_classes=NUM_CLASSES
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        with torch.no_grad():
            pred_probs = model(X_tensor)
            pred_class = torch.argmax(pred_probs, dim=1).item()
            confidence = torch.max(pred_probs, dim=1).values.item()
            
            # 根据分类结果计算预测价格
            last_close = df.iloc[-1]['close']
            
            # 2分类：0=跌，1=涨
            if pred_class == 0:  # 跌
                # 根据置信度调整预测幅度
                pred_change = -0.003 * confidence  # 0.3% * 置信度
                pred_close = last_close * (1 + pred_change)
                direction = "跌"
            else:  # 涨
                # 根据置信度调整预测幅度
                pred_change = 0.003 * confidence  # 0.3% * 置信度
                pred_close = last_close * (1 + pred_change)
                direction = "涨"
            
            pred_change_ratio = (pred_close - last_close) / last_close
            pred_change_ratio_pct = pred_change_ratio * 100
            
            print(f"🔍 分类预测结果:")
            print(f"  预测方向: {direction}")
            print(f"  置信度: {confidence:.3f}")
            print(f"  变化幅度: {pred_change_ratio_pct:.3f}%")
            
            return {
                'direction': direction,
                'confidence': confidence,
                'change_ratio': pred_change_ratio,
                'predicted_close': pred_close,
                'current_close': last_close
            }
            
    except Exception as e:
        print(f"❌ 分类预测失败: {e}")
        return None

def predict_regression(X_tensor: torch.Tensor, df: pd.DataFrame):
    """回归模型预测"""
    try:
        # 加载回归模型
        model = LSTMModel(
            input_size=X_tensor.shape[2], 
            hidden_size=64, 
            num_layers=2, 
            num_classes=1
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        with torch.no_grad():
            pred_normalized = model(X_tensor).item()
        
        # 将归一化的预测值转换回实际变化率
        pred_change_ratio = (pred_normalized * 2 * MAX_CHANGE_RATIO) - MAX_CHANGE_RATIO
        
        # 计算预测价格
        last_close = df.iloc[-1]['close']
        pred_close = last_close * (1 + pred_change_ratio)
        
        # 确定方向
        if pred_change_ratio > 0:
            direction = "涨"
        elif pred_change_ratio < 0:
            direction = "跌"
        else:
            direction = "平"
        
        pred_change_ratio_pct = pred_change_ratio * 100
        
        print(f"🔍 回归预测结果:")
        print(f"  预测方向: {direction}")
        print(f"  变化幅度: {pred_change_ratio_pct:.3f}%")
        print(f"  预测价格: {pred_close:.2f}")
        
        return {
            'direction': direction,
            'change_ratio': pred_change_ratio,
            'predicted_close': pred_close,
            'current_close': last_close
        }
        
    except Exception as e:
        print(f"❌ 回归预测失败: {e}")
        return None

def analyze_prediction_reason(df: pd.DataFrame, change_ratio: float, direction: str):
    """分析预测原因"""
    print(f"\n=== 预测原因分析 ===")
    
    last_row = df.iloc[-1]
    
    # RSI分析
    if 'rsi_14' in last_row:
        rsi = last_row['rsi_14']
        if rsi > 70:
            rsi_signal = "超买"
        elif rsi < 30:
            rsi_signal = "超卖"
        else:
            rsi_signal = "中性"
        print(f"RSI(14): {rsi:.1f} - {rsi_signal}")
    
    # 布林带分析
    if 'bb_position' in last_row:
        bb_pos = last_row['bb_position']
        if bb_pos > 0.8:
            bb_signal = "接近上轨"
        elif bb_pos < 0.2:
            bb_signal = "接近下轨"
        else:
            bb_signal = "中轨附近"
        print(f"布林带位置: {bb_pos:.2f} - {bb_signal}")
    
    # MACD分析
    if 'macd_histogram' in last_row:
        macd = last_row['macd_histogram']
        if macd > 0:
            macd_signal = "多头信号"
        else:
            macd_signal = "空头信号"
        print(f"MACD柱状图: {macd:.6f} - {macd_signal}")
    
    # 成交量分析
    if 'volume_ma_ratio' in last_row:
        vol_ratio = last_row['volume_ma_ratio']
        if vol_ratio > 1.5:
            vol_signal = "放量"
        elif vol_ratio < 0.5:
            vol_signal = "缩量"
        else:
            vol_signal = "正常"
        print(f"成交量比率: {vol_ratio:.2f} - {vol_signal}")
    
    # 动量分析
    if 'momentum_5' in last_row:
        momentum = last_row['momentum_5']
        if momentum > 0.02:
            mom_signal = "强势上涨"
        elif momentum < -0.02:
            mom_signal = "强势下跌"
        else:
            mom_signal = "震荡"
        print(f"5日动量: {momentum:.3f} - {mom_signal}")

def get_overall_analysis(change_ratio: float, direction: str, last_row: pd.Series):
    """综合分析"""
    print(f"\n=== 综合分析 ===")
    
    # 根据预测方向给出建议
    if direction == "涨":
        print("📈 看涨信号:")
        print("  - 建议关注买入机会")
        print("  - 设置止损位保护利润")
    elif direction == "跌":
        print("📉 看跌信号:")
        print("  - 建议谨慎操作")
        print("  - 可以考虑减仓或观望")
    else:
        print("➡️ 震荡信号:")
        print("  - 市场可能横盘整理")
        print("  - 建议等待明确方向")
    
    # 风险评估
    risk_level = "中等"
    if abs(change_ratio) > 0.01:  # 超过1%
        risk_level = "较高"
    elif abs(change_ratio) < 0.002:  # 小于0.2%
        risk_level = "较低"
    
    print(f"风险评估: {risk_level}")

def main():
    """主函数"""
    # 加载最新数据
    df = load_and_preprocess(DATA_PATH)
    
    if df is None or len(df) < WINDOW_SIZE + 1:
        print("❌ 数据不足，无法进行预测")
        return
    
    # 进行预测
    result = predict_next_candle_improved(df)
    
    if result is None:
        print("❌ 预测失败")
        return
    
    # 分析预测原因
    analyze_prediction_reason(df, result['change_ratio'], result['direction'])
    
    # 综合分析
    get_overall_analysis(result['change_ratio'], result['direction'], df.iloc[-1])
    
    print(f"\n✅ 预测完成！")

if __name__ == "__main__":
    main() 