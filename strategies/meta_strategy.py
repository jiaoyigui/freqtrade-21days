# -*- coding: utf-8 -*-
# Source: day14.md - MetaStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import pandas as pd


class MetaStrategy(IStrategy):
    """
    元策略：组合多个子策略的信号

    思路：
    - 每个子策略独立产生信号和置信度
    - 元策略根据权重加权投票
    - 超过阈值才开仓
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.01}
    stoploss = -0.05
    timeframe = '1h'

    # 子策略权重（可通过 hyperopt 优化）
    weight_mean_reversion = 0.3
    weight_trend = 0.35
    weight_breakout = 0.35

    # 信号阈值
    signal_threshold = 0.5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # === 均值回归指标 ===
        dataframe['bb_mid'] = dataframe['close'].rolling(20).mean()
        dataframe['bb_std'] = dataframe['close'].rolling(20).std()
        dataframe['bb_lower'] = dataframe['bb_mid'] - 2 * dataframe['bb_std']
        dataframe['bb_upper'] = dataframe['bb_mid'] + 2 * dataframe['bb_std']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_mid']
        dataframe['zscore'] = (dataframe['close'] - dataframe['bb_mid']) / dataframe['bb_std']

        # === 趋势跟随指标 ===
        dataframe['ema_fast'] = dataframe['close'].ewm(span=12).mean()
        dataframe['ema_slow'] = dataframe['close'].ewm(span=26).mean()
        dataframe['macd'] = dataframe['ema_fast'] - dataframe['ema_slow']
        dataframe['macd_signal'] = dataframe['macd'].ewm(span=9).mean()

        # ADX
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        dataframe['atr_14'] = tr.rolling(14).mean()

        plus_dm = np.where((high - high.shift(1)) > (low.shift(1) - low),
                           np.maximum(high - high.shift(1), 0), 0)
        minus_dm = np.where((low.shift(1) - low) > (high - high.shift(1)),
                            np.maximum(low.shift(1) - low, 0), 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / dataframe['atr_14']
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / dataframe['atr_14']
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dataframe['adx'] = dx.rolling(14).mean()

        # === 突破指标 ===
        dataframe['dc_upper'] = dataframe['high'].rolling(20).max()
        dataframe['dc_lower'] = dataframe['low'].rolling(20).min()
        dataframe['dc_mid'] = (dataframe['dc_upper'] + dataframe['dc_lower']) / 2

        dataframe['atr_ratio'] = dataframe['atr_14'] / dataframe['atr_14'].rolling(50).mean()

        # === 子策略信号 ===
        dataframe['signal_mr'] = (
            (dataframe['zscore'] < -2) &
            (dataframe['atr_ratio'] < 1.2)
        ).astype(float)

        dataframe['signal_trend'] = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1)) &
            (dataframe['adx'] > 25)
        ).astype(float)

        dataframe['signal_breakout'] = (
            (dataframe['close'] > dataframe['dc_upper'].shift(1)) &
            (dataframe['atr_ratio'] > 1.3)
        ).astype(float)

        # 加权综合得分
        dataframe['meta_score'] = (
            self.weight_mean_reversion * dataframe['signal_mr'] +
            self.weight_trend * dataframe['signal_trend'] +
            self.weight_breakout * dataframe['signal_breakout']
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['meta_score'] >= self.signal_threshold) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'meta_signal')

        # 标记信号来源
        for idx in dataframe.index[dataframe['enter_long'] == 1]:
            sources = []
            if dataframe.loc[idx, 'signal_mr'] > 0:
                sources.append('MR')
            if dataframe.loc[idx, 'signal_trend'] > 0:
                sources.append('TF')
            if dataframe.loc[idx, 'signal_breakout'] > 0:
                sources.append('BO')
            dataframe.loc[idx, 'enter_tag'] = f"meta_{'_'.join(sources)}"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['zscore'] > 2) |
            ((dataframe['macd'] < dataframe['macd_signal']) & (dataframe['adx'] > 20)),
            'exit_long'
        ] = 1
        return dataframe
