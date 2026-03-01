# -*- coding: utf-8 -*-
# Source: day16.md - Utility functions
# Freqtrade 21 天从入门到精通

import numpy as np
import pandas as pd
from typing import Union


class AlphaOperators:
    """Alpha 101 基础算子库"""

    @staticmethod
    def rank(df: pd.DataFrame) -> pd.DataFrame:
        """截面排名（百分位）"""
        return df.rank(axis=1, pct=True)

    @staticmethod
    def ts_rank(series: pd.Series, window: int) -> pd.Series:
        """时序排名：当前值在过去 window 期中的排名百分位"""
        return series.rolling(window).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )

    @staticmethod
    def ts_max(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).max()

    @staticmethod
    def ts_min(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).min()

    @staticmethod
    def ts_argmax(series: pd.Series, window: int) -> pd.Series:
        """过去 window 期最大值出现的位置"""
        return series.rolling(window).apply(lambda x: x.argmax(), raw=True)

    @staticmethod
    def ts_argmin(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).apply(lambda x: x.argmin(), raw=True)

    @staticmethod
    def delta(series: pd.Series, period: int = 1) -> pd.Series:
        """差分"""
        return series.diff(period)

    @staticmethod
    def delay(series: pd.Series, period: int = 1) -> pd.Series:
        """延迟"""
        return series.shift(period)

    @staticmethod
    def correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """滚动相关系数"""
        return x.rolling(window).corr(y)

    @staticmethod
    def covariance(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
        """滚动协方差"""
        return x.rolling(window).cov(y)

    @staticmethod
    def scale(series: pd.Series, a: float = 1.0) -> pd.Series:
        """缩放到绝对值之和为 a"""
        return series * a / series.abs().sum()

    @staticmethod
    def decay_linear(series: pd.Series, window: int) -> pd.Series:
        """线性衰减加权移动平均"""
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()
        return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

    @staticmethod
    def stddev(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).std()

    @staticmethod
    def product(series: pd.Series, window: int) -> pd.Series:
        """滚动乘积"""
        return series.rolling(window).apply(lambda x: np.prod(x), raw=True)

    @staticmethod
    def sum_(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).sum()


# Alpha 因子函数
def alpha_001(close: pd.Series, returns: pd.Series) -> pd.Series:
    """Alpha#1: 条件波动率/价格的极值位置"""
    op = AlphaOperators()
    inner = np.where(returns < 0, op.stddev(returns, 20), close)
    inner = pd.Series(inner, index=close.index)
    signed_power = np.sign(inner) * (inner.abs() ** 2)
    return op.ts_argmax(signed_power, 5) / 5 - 0.5


def alpha_006(open_: pd.Series, volume: pd.Series) -> pd.Series:
    """Alpha#6: 开盘价与成交量的负相关"""
    return -1 * AlphaOperators.correlation(open_, volume, 10)


def alpha_012(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Alpha#12: 量价动量背离"""
    return np.sign(volume.diff(1)) * (-1 * close.diff(1))


def alpha_021(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Alpha#21: 综合均值回归"""
    mean_8 = close.rolling(8).mean()
    std_8 = close.rolling(8).std()
    mean_2 = close.rolling(2).mean()

    cond1 = (mean_8 + std_8) < mean_8
    cond2 = mean_8 - std_8 > mean_2
    cond3 = volume.rolling(20).mean() / volume

    result = np.where(cond1, -1,
             np.where(cond2, 1,
             np.where(cond3 >= 1, -1, 1)))

    return pd.Series(result, index=close.index).astype(float)


def alpha_033(close: pd.Series, open_: pd.Series) -> pd.Series:
    """Alpha#33: rank(-1 * (1 - open/close))"""
    return (-1 * (1 - open_ / close)).rank(pct=True)


def alpha_041(high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
    """Alpha#41: power(high * low, 0.5) - VWAP 近似"""
    hl_mean = np.sqrt(high * low)
    return hl_mean.rank(pct=True) * (-1 * volume.diff(1).rank(pct=True))


def alpha_053(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Alpha#53: 价格在近期通道中的位置变化"""
    inner = (close - low - (high - close)) / (close - low + 1e-8)
    return -1 * inner.diff(9)


def alpha_054(open_: pd.Series, close: pd.Series,
              high: pd.Series, low: pd.Series) -> pd.Series:
    """Alpha#54: 价格形态因子"""
    numerator = -1 * (low - close) * (open_ ** 5)
    denominator = (low - high + 1e-8) * (close ** 5)
    return numerator / denominator


def alpha_085(close: pd.Series, volume: pd.Series, high: pd.Series) -> pd.Series:
    """Alpha#85: 加权价格与平均成交量的相关性排名"""
    weighted_price = high * 0.876 + close * 0.124
    avg_vol = volume.rolling(30).mean()
    corr = AlphaOperators.correlation(weighted_price, avg_vol, 10)
    return corr.rank(pct=True)


def alpha_101(close: pd.Series, open_: pd.Series,
              high: pd.Series, low: pd.Series) -> pd.Series:
    """Alpha#101: K 线实体占比"""
    return (close - open_) / (high - low + 0.001)


def factor_decay_analysis(ic_series: pd.Series, window: int = 60) -> dict:
    """检测因子是否在衰减"""
    from scipy import stats

    rolling_ic = ic_series.rolling(window).mean()
    x = np.arange(len(rolling_ic.dropna()))
    y = rolling_ic.dropna().values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        'slope': slope,
        'p_value': p_value,
        'is_decaying': slope < 0 and p_value < 0.05,
        'half_life': abs(rolling_ic.mean() / slope) if slope != 0 else float('inf'),
        'current_ic': rolling_ic.iloc[-1] if len(rolling_ic) > 0 else None
    }
