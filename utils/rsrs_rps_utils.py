# -*- coding: utf-8 -*-
# Source: day15.md - Utility functions
# Freqtrade 21 天从入门到精通

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rsrs_right_skew(beta_series: pd.Series, window: int = 600) -> pd.Series:
    """
    右偏标准化 RSRS
    对标准化后的 RSRS 再乘以自身的绝对值，放大极端值
    """
    mu = beta_series.rolling(window).mean()
    sigma = beta_series.rolling(window).std()
    std_rsrs = (beta_series - mu) / sigma
    return std_rsrs * abs(std_rsrs)


def calculate_rps(close_prices: pd.DataFrame, periods: list = [20, 60, 120]) -> pd.DataFrame:
    """
    计算 RPS 相对价格强度

    Args:
        close_prices: DataFrame，每列是一个交易对的收盘价
        periods: 计算动量的周期列表

    Returns:
        RPS 排名百分位（0-100），100 = 最强
    """
    rps_scores = pd.DataFrame(index=close_prices.index)

    for period in periods:
        returns = close_prices.pct_change(period)
        rps = returns.rank(axis=1, pct=True) * 100

        for col in close_prices.columns:
            rps_scores[f'{col}_rps_{period}'] = rps[col]

    for col in close_prices.columns:
        rps_cols = [f'{col}_rps_{p}' for p in periods]
        weights = [0.5, 0.3, 0.2]
        rps_scores[f'{col}_rps_composite'] = sum(
            w * rps_scores[c] for w, c in zip(weights, rps_cols)
        )

    return rps_scores


def factor_ic_analysis(factor_values: pd.DataFrame,
                       forward_returns: pd.DataFrame,
                       periods: list = [1, 5, 20]) -> pd.DataFrame:
    """
    计算因子的 IC（Rank Information Coefficient）

    IC = Spearman 相关系数 (因子排名，未来收益排名)
    |IC| > 0.03 且稳定 → 因子有效
    """
    results = {}
    for period in periods:
        ic_series = []
        for date in factor_values.index:
            if date not in forward_returns.index:
                continue
            fv = factor_values.loc[date].dropna()
            fr = forward_returns.loc[date].dropna()
            common = fv.index.intersection(fr.index)
            if len(common) < 5:
                continue
            ic, _ = spearmanr(fv[common], fr[common])
            ic_series.append(ic)

        ic_arr = np.array(ic_series)
        results[f'{period}d'] = {
            'IC_mean': np.mean(ic_arr),
            'IC_std': np.std(ic_arr),
            'ICIR': np.mean(ic_arr) / np.std(ic_arr) if np.std(ic_arr) > 0 else 0,
            'IC_positive_ratio': np.mean(ic_arr > 0)
        }

    return pd.DataFrame(results).T


def layered_backtest(factor_values: pd.Series, returns: pd.Series,
                     n_groups: int = 5) -> dict:
    """
    分层回测：按因子值分成 N 组，看各组收益是否单调递增
    如果 Top 组显著跑赢 Bottom 组 → 因子有效
    """
    groups = pd.qcut(factor_values, n_groups, labels=False, duplicates='drop')

    group_returns = {}
    for g in range(n_groups):
        mask = groups == g
        group_returns[f'G{g+1}'] = returns[mask].mean()

    long_short = group_returns[f'G{n_groups}'] - group_returns['G1']

    return {
        'group_returns': group_returns,
        'long_short': long_short,
        'monotonic': all(
            group_returns[f'G{i+1}'] <= group_returns[f'G{i+2}']
            for i in range(n_groups - 1)
        )
    }
