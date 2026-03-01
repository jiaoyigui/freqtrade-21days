# -*- coding: utf-8 -*-
# Source: day12.md - FreqAIRobustStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta


class FreqAIRobustStrategy(IStrategy):
    """
    稳健的 FreqAI 策略

    设计原则：
    1. 特征要有金融学直觉，不是随机组合
    2. 目标变量要合理（不要预测精确价格）
    3. 严格依赖 do_predict 过滤不可信预测
    4. 传统指标作为安全网（ML 失效时兜底）
    """
    timeframe = '1h'
    stoploss = -0.06
    use_custom_stoploss = True

    minimal_roi = {"0": 0.08, "120": 0.04, "360": 0.01}

    # FreqAI 配置
    freqai_info = {"enabled": True}

    def feature_engineering_expand_all(self, dataframe, period, metadata, **kwargs):
        """按 indicator_periods_candles 展开的特征"""
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-volatility-period"] = dataframe['close'].pct_change().rolling(period).std()
        dataframe["%-momentum-period"] = dataframe['close'].pct_change(period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
        """不需要周期展开的特征"""
        dataframe["%-close_to_high"] = dataframe['close'] / dataframe['high'].rolling(20).max()
        dataframe["%-close_to_low"] = dataframe['close'] / dataframe['low'].rolling(20).min()
        dataframe["%-volume_ratio"] = dataframe['volume'] / dataframe['volume'].rolling(50).mean()

        body = abs(dataframe['close'] - dataframe['open'])
        total = dataframe['high'] - dataframe['low']
        dataframe["%-body_ratio"] = body / total.replace(0, np.nan)

        dataframe["%-hour_sin"] = np.sin(2 * np.pi * dataframe['date'].dt.hour / 24)
        dataframe["%-hour_cos"] = np.cos(2 * np.pi * dataframe['date'].dt.hour / 24)
        return dataframe

    def set_freqai_targets(self, dataframe, metadata, **kwargs):
        """定义预测目标"""
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-future_return"] = (
            dataframe["close"].shift(-label_period) / dataframe["close"] - 1
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["&-future_return"] > 0.02) &
                (dataframe["do_predict"] == 1) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['rsi'] < 70) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'freqai_bullish')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["&-future_return"] < -0.005) |
                (dataframe["do_predict"] != 1) |
                (dataframe['rsi'] > 78)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'freqai_exit')

        return dataframe
