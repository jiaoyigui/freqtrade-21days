# -*- coding: utf-8 -*-
# Source: day06.md - Utility functions
# Freqtrade 21 天从入门到精通

import numpy as np
import talib.abstract as ta


def demonstrate_filter_properties(dataframe):
    """展示不同均线的滤波特性"""
    close = dataframe['close']

    # 不同窗口长度 = 不同截止频率
    sma_5 = ta.SMA(dataframe, timeperiod=5)
    sma_20 = ta.SMA(dataframe, timeperiod=20)
    sma_50 = ta.SMA(dataframe, timeperiod=50)

    # EMA 是 IIR（无限脉冲响应）滤波器
    ema_20 = ta.EMA(dataframe, timeperiod=20)

    return sma_5, sma_20, sma_50, ema_20


def rsi_from_scratch(close, period=14):
    """手写 RSI，理解每一步"""
    delta = close.diff()

    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    # Wilder 的平滑方法（等价于 EMA，alpha = 1/period）
    avg_gain = gains.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def kaufman_ama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    核心思想：市场趋势强时用快均线，震荡时用慢均线
    """
    # 效率比（Efficiency Ratio）
    direction = abs(close - close.shift(er_period))
    volatility = close.diff().abs().rolling(er_period).sum()
    er = direction / volatility

    # 自适应平滑系数
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # 递归计算
    ama = close.copy()
    for i in range(er_period, len(close)):
        ama.iloc[i] = ama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - ama.iloc[i-1])

    return ama


def hp_decompose(close, lamb=1600):
    """
    HP 滤波器：把价格分解为趋势 + 周期成分

    lamb 参数控制平滑度：
    - lamb=6.25: 适合年度数据
    - lamb=1600: 适合季度数据（默认）
    - lamb=129600: 适合月度数据
    对于 15 分钟 K 线，建议 lamb=10000-50000
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle, trend = hpfilter(close, lamb=lamb)
    return trend, cycle


def wavelet_denoise(close, wavelet='db4', level=3):
    """
    小波去噪：比移动平均更智能的滤波方法

    优势：在去噪的同时保留价格的突变特征
    """
    import pywt

    coeffs = pywt.wavedec(close, wavelet, level=level)

    # 对高频系数做软阈值处理
    threshold = np.sqrt(2 * np.log(len(close))) * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]

    denoised = pywt.waverec(coeffs, wavelet)
    return denoised[:len(close)]
