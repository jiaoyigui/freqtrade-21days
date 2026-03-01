# -*- coding: utf-8 -*-
# Source: day04.md - Utility functions
# Freqtrade 21 天从入门到精通

import pandas as pd
import numpy as np


def check_data_gaps(dataframe, timeframe_minutes=15):
    """检测数据缺口"""
    expected_delta = pd.Timedelta(minutes=timeframe_minutes)
    time_diffs = dataframe['date'].diff()
    gaps = dataframe[time_diffs > expected_delta * 1.5]

    if len(gaps) > 0:
        print(f"⚠️ 发现 {len(gaps)} 个数据缺口：")
        for _, row in gaps.iterrows():
            print(f"  {row['date']} (间隔 {time_diffs.loc[row.name]})")
    else:
        print("✅ 数据连续，无缺口")

    total_expected = (dataframe['date'].max() - dataframe['date'].min()) / expected_delta
    actual = len(dataframe)
    completeness = actual / total_expected * 100
    print(f"数据完整率：{completeness:.1f}% ({actual}/{int(total_expected)})")
    return gaps


def detect_outliers(dataframe, z_threshold=5):
    """用 Z-score 检测价格异常值"""
    returns = dataframe['close'].pct_change()
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = dataframe[z_scores.abs() > z_threshold]

    if len(outliers) > 0:
        print(f"⚠️ 发现 {len(outliers)} 个异常 K 线（Z-score > {z_threshold}）：")
        for _, row in outliers.iterrows():
            ret = returns.loc[row.name]
            z = z_scores.loc[row.name]
            print(f"  {row['date']}: 涨跌幅 {ret:.2%}, Z-score {z:.1f}")
    else:
        print("✅ 未发现异常值")
    return outliers


def detect_wicks(dataframe, wick_ratio=3.0):
    """检测异常长影线（可能是插针）"""
    body = abs(dataframe['close'] - dataframe['open'])
    upper_wick = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
    lower_wick = dataframe[['open', 'close']].min(axis=1) - dataframe['low']

    abnormal = dataframe[
        (upper_wick > body * wick_ratio) | (lower_wick > body * wick_ratio)
    ]
    print(f"发现 {len(abnormal)} 根异常长影线 K 线（影线 > 实体 × {wick_ratio}）")
    return abnormal


def full_data_audit(dataframe, pair, timeframe_minutes=15):
    """一键数据质量审计"""
    print(f"\n{'='*50}")
    print(f"数据审计报告：{pair}")
    print(f"{'='*50}")
    print(f"时间范围：{dataframe['date'].min()} → {dataframe['date'].max()}")
    print(f"K 线数量：{len(dataframe)}")
    print()

    # 1. 缺口检测
    print("--- 缺口检测 ---")
    gaps = check_data_gaps(dataframe, timeframe_minutes)
    print()

    # 2. 异常值检测
    print("--- 异常值检测 ---")
    outliers = detect_outliers(dataframe)
    print()

    # 3. 基础统计
    print("--- 基础统计 ---")
    returns = dataframe['close'].pct_change().dropna()
    print(f"  日均波动率：{returns.std():.4f} ({returns.std()*100:.2f}%)")
    print(f"  最大单 K 涨幅：{returns.max():.2%}")
    print(f"  最大单 K 跌幅：{returns.min():.2%}")
    print(f"  零成交量 K 线：{(dataframe['volume'] == 0).sum()}")

    return {'gaps': len(gaps), 'outliers': len(outliers)}
