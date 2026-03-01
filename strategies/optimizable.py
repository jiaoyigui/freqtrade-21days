# -*- coding: utf-8 -*-
# Source: day20.md - OptimizableStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame


class OptimizableStrategy(IStrategy):
    """
    可优化策略模板

    关键原则：
    1. 参数空间不要太大（维度诅咒）
    2. 每个参数要有物理意义
    3. 参数之间尽量独立
    """

    INTERFACE_VERSION = 3
    timeframe = '1h'

    # 入场参数
    buy_rsi = IntParameter(20, 40, default=30, space='buy')
    buy_ema_short = IntParameter(5, 20, default=10, space='buy')
    buy_ema_long = IntParameter(20, 60, default=50, space='buy')
    buy_volume_factor = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space='buy')

    # 出场参数
    sell_rsi = IntParameter(60, 85, default=70, space='sell')
    sell_profit_target = DecimalParameter(0.02, 0.10, default=0.05, decimals=2, space='sell')

    minimal_roi = {
        "0": 0.08,
        "30": 0.04,
        "60": 0.02,
        "120": 0.01
    }

    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI
        delta = dataframe['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        # 动态 EMA
        for period in range(5, 61):
            dataframe[f'ema_{period}'] = dataframe['close'].ewm(span=period).mean()

        dataframe['vol_ma'] = dataframe['volume'].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_short = dataframe[f'ema_{self.buy_ema_short.value}']
        ema_long = dataframe[f'ema_{self.buy_ema_long.value}']

        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value) &
                (ema_short > ema_long) &
                (dataframe['volume'] > dataframe['vol_ma'] * self.buy_volume_factor.value) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'optimized_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > self.sell_rsi.value),
            'exit_long'
        ] = 1
        return dataframe
