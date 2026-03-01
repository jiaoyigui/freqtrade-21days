# -*- coding: utf-8 -*-
# Source: day07.md - Utility functions
# Freqtrade 21 天从入门到精通

import numpy as np
import pandas as pd


def kelly_fraction(returns):
    """
    计算连续凯利比例

    Args:
        returns: 策略的历史收益率序列

    Returns:
        包含全凯利、半凯利等信息的字典
    """
    mu = np.mean(returns)
    sigma2 = np.var(returns)

    if sigma2 == 0:
        return 0

    full_kelly = mu / sigma2
    half_kelly = full_kelly / 2

    return {
        'full_kelly': full_kelly,
        'half_kelly': half_kelly,
        'expected_return': mu,
        'variance': sigma2,
        'sharpe': mu / np.sqrt(sigma2) if sigma2 > 0 else 0
    }


def ruin_probability(win_rate, payoff_ratio, risk_per_trade, initial_capital=1.0):
    """
    蒙特卡洛模拟破产概率

    Args:
        win_rate: 胜率
        payoff_ratio: 盈亏比
        risk_per_trade: 每笔风险比例
        initial_capital: 初始资金
    """
    n_simulations = 10000
    n_trades = 1000
    ruin_count = 0
    ruin_threshold = 0.1

    for _ in range(n_simulations):
        capital = initial_capital
        for _ in range(n_trades):
            if np.random.random() < win_rate:
                capital += capital * risk_per_trade * payoff_ratio
            else:
                capital -= capital * risk_per_trade

            if capital < initial_capital * ruin_threshold:
                ruin_count += 1
                break

    return ruin_count / n_simulations


def expected_max_drawdown(sharpe_ratio, n_periods):
    """
    估计期望最大回撤

    基于 Magdon-Ismail et al. (2004) 的近似公式
    """
    if sharpe_ratio <= 0:
        return float('inf')

    expected_mdd = np.sqrt(np.pi / 8) * np.sqrt(n_periods) / sharpe_ratio
    return expected_mdd


def calculate_risk_metrics(returns, confidence=0.95):
    """计算 VaR 和 CVaR"""
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)

    var_index = int(n * (1 - confidence))
    var = -sorted_returns[var_index]
    cvar = -np.mean(sorted_returns[:var_index])

    return {
        'VaR_95': var,
        'CVaR_95': cvar,
        'max_loss': -sorted_returns[0],
        'mean_return': np.mean(returns),
        'std': np.std(returns),
        'skewness': float(pd.Series(returns).skew()),
        'kurtosis': float(pd.Series(returns).kurtosis())
    }
