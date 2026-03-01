# -*- coding: utf-8 -*-
# Source: day08.md - Utility functions
# Freqtrade 21 天从入门到精通

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm


def estimate_ou_parameters(prices):
    """
    估计 OU 过程的参数
    使用 AR(1) 回归：X(t) - X(t-1) = a + b*X(t-1) + ε
    """
    X = np.array(prices)
    dX = np.diff(X)
    X_lag = X[:-1]

    # OLS 回归
    X_lag_with_const = np.column_stack([np.ones(len(X_lag)), X_lag])
    model = OLS(dX, X_lag_with_const).fit()

    a, b = model.params

    # OU 参数
    theta = -b
    mu = a / theta if theta != 0 else 0
    sigma = np.std(model.resid)
    half_life = np.log(2) / theta if theta != 0 else float('inf')

    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life': half_life,
        'r_squared': model.rsquared
    }


def adf_test(prices):
    """ADF 检验：价格序列是否平稳（均值回归）"""
    result = adfuller(prices, maxlag=20, regression='c')
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05,
        'critical_values': result[4]
    }


def find_cointegrated_pairs(price_data, significance=0.05):
    """
    寻找协整的交易对

    协整 vs 相关：
    - 相关：两个序列同涨同跌（短期关系）
    - 协整：两个序列的线性组合是平稳的（长期均衡关系）
    """
    pairs = []
    symbols = list(price_data.keys())
    n = len(symbols)

    for i in range(n):
        for j in range(i+1, n):
            s1 = price_data[symbols[i]]
            s2 = price_data[symbols[j]]

            score, pvalue, _ = coint(s1, s2)

            if pvalue < significance:
                model = sm.OLS(s1, sm.add_constant(s2)).fit()
                hedge_ratio = model.params[1]

                spread = s1 - hedge_ratio * s2
                ou_params = estimate_ou_parameters(spread)

                pairs.append({
                    'pair': (symbols[i], symbols[j]),
                    'p_value': pvalue,
                    'hedge_ratio': hedge_ratio,
                    'half_life': ou_params['half_life'],
                    'mean': ou_params['mu'],
                    'theta': ou_params['theta']
                })

    return sorted(pairs, key=lambda x: x['p_value'])
