# -*- coding: utf-8 -*-
# Source: day05.md - Utility functions
# Freqtrade 21 天从入门到精通

import numpy as np
from scipy.stats import norm


def deflated_sharpe_ratio(sharpe_ratio, num_trials, variance_of_sharpe, T):
    """
    Bailey & López de Prado (2014) 的 Deflated Sharpe Ratio

    Args:
        sharpe_ratio: 最佳策略的夏普比率
        num_trials: 尝试了多少个策略变体
        variance_of_sharpe: 夏普比率的方差
        T: 回测的观测数量

    Returns:
        DSR > 0.95 才算统计显著
    """
    expected_max_sr = np.sqrt(variance_of_sharpe) * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1/num_trials)
        + np.euler_gamma * norm.ppf(1 - 1/(num_trials * np.e))
    )

    dsr = norm.cdf(
        (sharpe_ratio - expected_max_sr) * np.sqrt(T - 1)
        / np.sqrt(1 - sharpe_ratio * np.sqrt(1/T) + sharpe_ratio**2 / (4*T))
    )

    return dsr


def min_trades_for_significance(sharpe_ratio, confidence=0.95):
    """计算达到统计显著性所需的最少交易数"""
    z = norm.ppf(confidence)
    min_T = (z / sharpe_ratio) ** 2
    return int(np.ceil(min_T))


def split_data(dataframe, train_ratio=0.6, val_ratio=0.2):
    """数据集划分"""
    n = len(dataframe)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = dataframe.iloc[:train_end]
    val = dataframe.iloc[train_end:val_end]
    test = dataframe.iloc[val_end:]

    print(f"训练集：{train['date'].min()} → {train['date'].max()} ({len(train)} 根 K 线)")
    print(f"验证集：{val['date'].min()} → {val['date'].max()} ({len(val)} 根 K 线)")
    print(f"测试集：{test['date'].min()} → {test['date'].max()} ({len(test)} 根 K 线)")

    return train, val, test


def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni 校正：最简单但最保守"""
    n = len(p_values)
    adjusted_alpha = alpha / n
    significant = [p < adjusted_alpha for p in p_values]

    print(f"测试了 {n} 个策略")
    print(f"原始显著性水平：{alpha}")
    print(f"校正后显著性水平：{adjusted_alpha:.6f}")
    print(f"通过校正的策略数：{sum(significant)}/{n}")

    return significant, adjusted_alpha
