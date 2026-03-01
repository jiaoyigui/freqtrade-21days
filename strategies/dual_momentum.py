# -*- coding: utf-8 -*-
# Source: day09.md - DualMomentumStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import numpy as np
import talib.abstract as ta


class DualMomentumStrategy(IStrategy):
    """
    双动量策略

    结合绝对动量（时间序列）和相对动量（横截面）：
    1. 绝对动量：资产自身过去 N 期收益 > 0 → 做多信号
    2. 相对动量：在多个资产中选动量最强的

    参考：Antonacci, "Dual momentum Investing" (2014)
    """
    timeframe = '4h'
    stoploss = -0.08

    momentum_period = IntParameter(30, 120, default=60, space='buy')

    minimal_roi = {"0": 0.15, "360": 0.08, "720": 0.03}

    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.06
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        period = self.momentum_period.value

        # 绝对动量：过去 N 期的收益率
        dataframe['momentum'] = dataframe['close'] / dataframe['close'].shift(period) - 1

        # 动量的加速度（动量的变化率）
        dataframe['momentum_accel'] = dataframe['momentum'] - dataframe['momentum'].shift(10)

        # 波动率标准化的动量（风险调整动量）
        returns = dataframe['close'].pct_change()
        vol = returns.rolling(period).std()
        dataframe['risk_adj_momentum'] = dataframe['momentum'] / (vol * np.sqrt(period))

        # 趋势强度
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # 均线系统
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)

        # 成交量动量
        dataframe['volume_momentum'] = (
            dataframe['volume'].rolling(10).mean()
            / dataframe['volume'].rolling(50).mean()
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['momentum'] > 0) &
                (dataframe['momentum_accel'] > 0) &
                (dataframe['risk_adj_momentum'] > 0.5) &
                (dataframe['adx'] > 20) &
                (dataframe['ema_20'] > dataframe['ema_50']) &
                (dataframe['volume_momentum'] > 1.0) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'dual_momentum_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['momentum'] < 0) |
                (
                    (dataframe['ema_20'] < dataframe['ema_50']) &
                    (dataframe['ema_20'].shift(1) >= dataframe['ema_50'].shift(1))
                )
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'momentum_reversal')

        return dataframe
