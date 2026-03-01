# -*- coding: utf-8 -*-
# Source: day11.md - Utility functions
# Freqtrade 21 天从入门到精通

import pandas as pd
import numpy as np
from numpy.linalg import qr


def build_momentum_factor(price_data, lookback=30, skip=1):
    """
    构建动量因子

    Args:
        price_data: dict {symbol: pd.Series of prices}
        lookback: 回看期（天）
        skip: 跳过最近 N 天（避免短期反转效应）

    Returns:
        每个时间点每个币的动量因子值
    """
    momentum = {}
    for symbol, prices in price_data.items():
        ret = prices.shift(skip) / prices.shift(lookback + skip) - 1
        momentum[symbol] = ret

    df = pd.DataFrame(momentum)
    df_zscore = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return df_zscore


def build_volatility_factor(price_data, window=30):
    """构建波动率因子（低波动率 = 高因子值）"""
    vol = {}
    for symbol, prices in price_data.items():
        returns = prices.pct_change()
        realized_vol = returns.rolling(window).std() * np.sqrt(365)
        vol[symbol] = -realized_vol

    df = pd.DataFrame(vol)
    df_zscore = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return df_zscore


def build_volume_factor(volume_data, marketcap_data):
    """构建流动性因子（换手率 = 成交量/市值）"""
    turnover = {}
    for symbol in volume_data:
        if symbol in marketcap_data:
            turnover[symbol] = volume_data[symbol] / marketcap_data[symbol]

    df = pd.DataFrame(turnover)
    df_zscore = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return df_zscore


def orthogonalize_factors(factor_dict):
    """
    Gram-Schmidt 正交化

    为什么要正交化？
    如果动量因子和波动率因子相关性 0.6，
    你以为你在用两个因子，其实 60% 的信息是重复的。
    正交化后，每个因子贡献独立的信息。
    """
    names = list(factor_dict.keys())
    aligned = pd.DataFrame(factor_dict).dropna()

    Q, R = qr(aligned.values)

    orthogonal = pd.DataFrame(Q[:, :len(names)],
                               index=aligned.index,
                               columns=[f"{n}_orth" for n in names])

    return orthogonal


def detect_factor_crowding(factor_returns, market_returns, window=60):
    """
    检测因子是否拥挤

    拥挤的信号：
    1. 因子收益与市场收益的相关性突然升高
    2. 因子收益的波动率突然升高
    3. 因子的夏普比率持续下降
    """
    rolling_corr = factor_returns.rolling(window).corr(market_returns)
    rolling_vol = factor_returns.rolling(window).std()
    rolling_sharpe = (factor_returns.rolling(window).mean() /
                      factor_returns.rolling(window).std())

    crowding_signal = (rolling_corr > 0.7) & (rolling_vol > rolling_vol.rolling(120).mean())

    return {
        'correlation': rolling_corr,
        'volatility': rolling_vol,
        'sharpe': rolling_sharpe,
        'is_crowded': crowding_signal
    }
