# -*- coding: utf-8 -*-
# Source: day08.md - BollingerMeanRevert
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class BollingerMeanRevert(IStrategy):
    """
    布林带均值回归策略

    理论基础：
    - 价格触及布林带下轨 = 短期超卖，统计上倾向回归均值
    - 加入成交量确认和趋势过滤，减少假信号
    - 用 Z-score 量化偏离程度，而不是简单的"触及上下轨"

    参考：Bollinger, "Bollinger on Bollinger Bands" (2001)
    """
    timeframe = '1h'
    stoploss = -0.06

    # 可优化参数
    bb_period = IntParameter(15, 30, default=20, space='buy')
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space='buy')
    zscore_entry = DecimalParameter(-3.0, -1.5, default=-2.0, decimals=1, space='buy')
    zscore_exit = DecimalParameter(-0.5, 0.5, default=0.0, decimals=1, space='sell')

    minimal_roi = {"0": 0.04, "60": 0.02, "180": 0.01}

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 布林带
        bb = ta.BBANDS(dataframe, timeperiod=self.bb_period.value,
                       nbdevup=self.bb_std.value, nbdevdn=self.bb_std.value)
        dataframe['bb_upper'] = bb['upperband']
        dataframe['bb_middle'] = bb['middleband']
        dataframe['bb_lower'] = bb['lowerband']

        # Z-score：标准化的偏离程度
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        rolling_mean = dataframe['close'].rolling(self.bb_period.value).mean()
        rolling_std = dataframe['close'].rolling(self.bb_period.value).std()
        dataframe['zscore'] = (dataframe['close'] - rolling_mean) / rolling_std

        # 趋势过滤
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # 成交量确认
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()

        # RSI 辅助确认
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['zscore'] < self.zscore_entry.value) &
                (dataframe['close'] > dataframe['ema_200'] * 0.92) &
                (dataframe['bb_width'] > 0.02) &
                (dataframe['volume'] > dataframe['volume_sma'] * 1.2) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'bb_zscore_oversold')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['zscore'] > self.zscore_exit.value) |
                (dataframe['rsi'] > 72)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'mean_reversion_complete')

        return dataframe
