# -*- coding: utf-8 -*-
# Source: day13.md - FundingRateStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class FundingRateStrategy(IStrategy):
    """
    资金费率监控策略（简化版）

    Freqtrade 不支持同时现货 + 合约，这里只做信号监控：
    当资金费率极端时，在现货市场做反向交易
    """
    timeframe = '1h'
    stoploss = -0.04
    minimal_roi = {"0": 0.03, "120": 0.015}

    def bot_loop_start(self, current_time, **kwargs):
        if self.dp.runmode.value in ('live', 'dry_run'):
            try:
                for pair in self.dp.current_whitelist():
                    # 实际实现需要通过 exchange API 获取 funding rate
                    pass
            except Exception:
                pass

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)

        dataframe['extreme_bullish'] = (dataframe['rsi'] > 80).astype(int)
        dataframe['extreme_bearish'] = (dataframe['rsi'] < 20).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['extreme_bearish'] == 1) &
                (dataframe['close'] > dataframe['ema_20'] * 0.95) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'funding_contrarian')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 65),
            'exit_long'] = 1
        return dataframe
