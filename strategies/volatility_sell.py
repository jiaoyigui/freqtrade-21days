# -*- coding: utf-8 -*-
# Source: day10.md - VolatilitySellStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta


class VolatilitySellStrategy(IStrategy):
    """
    卖波动率策略（波动率均值回归）

    逻辑：波动率飙升后倾向于回落
    当 VIX（或 ATR）极高时，市场恐慌过度，做多

    风险：如果波动率继续飙升（黑天鹅），亏损巨大
    → 必须严格止损

    参考：Ilmanen, "Expected Returns" (2011), Chapter 19
    """
    timeframe = '1h'
    stoploss = -0.05
    minimal_roi = {"0": 0.03, "120": 0.015, "360": 0.005}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 已实现波动率
        returns = dataframe['close'].pct_change()
        dataframe['realized_vol'] = returns.rolling(24).std() * np.sqrt(24 * 365)

        # 波动率的百分位排名
        dataframe['vol_percentile'] = dataframe['realized_vol'].rolling(500).rank(pct=True)

        # 波动率的变化率
        dataframe['vol_change'] = dataframe['realized_vol'].pct_change(6)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['vol_percentile'] > 0.90) &
                (dataframe['vol_change'] < 0) &
                (dataframe['rsi'] < 30) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'vol_mean_revert')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['vol_percentile'] < 0.50) |
                (dataframe['rsi'] > 65)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'vol_normalized')

        return dataframe
