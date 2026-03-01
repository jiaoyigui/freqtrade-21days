# -*- coding: utf-8 -*-
# Source: day16.md - Alpha101Strategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np

# 引入 Alpha 因子函数
from ..utils.alpha_operators import (
    alpha_001, alpha_006, alpha_012, alpha_021, alpha_033,
    alpha_041, alpha_053, alpha_054, alpha_085, alpha_101
)


class Alpha101Strategy(IStrategy):
    """
    Alpha 101 多因子组合策略

    选取 10 个适合加密市场的因子，等权组合
    通过因子得分的排名百分位产生信号
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.06, "30": 0.03, "60": 0.015}
    stoploss = -0.05
    timeframe = '1h'

    # 因子权重（可 hyperopt）
    factor_weights = {
        'alpha_001': 0.10,
        'alpha_006': 0.10,
        'alpha_012': 0.15,
        'alpha_021': 0.10,
        'alpha_033': 0.10,
        'alpha_041': 0.10,
        'alpha_053': 0.10,
        'alpha_054': 0.05,
        'alpha_085': 0.10,
        'alpha_101': 0.10,
    }

    entry_threshold = 0.7   # 综合得分 > 70% 分位
    exit_threshold = 0.3    # 综合得分 < 30% 分位

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        close = dataframe['close']
        open_ = dataframe['open']
        high = dataframe['high']
        low = dataframe['low']
        volume = dataframe['volume']
        returns = close.pct_change()

        # 计算各因子
        dataframe['alpha_001'] = alpha_001(close, returns)
        dataframe['alpha_006'] = alpha_006(open_, volume)
        dataframe['alpha_012'] = alpha_012(close, volume)
        dataframe['alpha_021'] = alpha_021(close, volume)
        dataframe['alpha_033'] = alpha_033(close, open_)
        dataframe['alpha_041'] = alpha_041(high, low, volume)
        dataframe['alpha_053'] = alpha_053(close, high, low)
        dataframe['alpha_054'] = alpha_054(open_, close, high, low)
        dataframe['alpha_085'] = alpha_085(close, volume, high)
        dataframe['alpha_101'] = alpha_101(close, open_, high, low)

        # 各因子标准化为排名百分位
        for name in self.factor_weights:
            col = dataframe[name]
            dataframe[f'{name}_rank'] = col.rolling(120).rank(pct=True)

        # 加权综合得分
        dataframe['composite_score'] = sum(
            self.factor_weights[name] * dataframe[f'{name}_rank']
            for name in self.factor_weights
        )

        # 综合得分的排名
        dataframe['score_percentile'] = dataframe['composite_score'].rolling(60).rank(pct=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['score_percentile'] > self.entry_threshold) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'alpha101_bullish')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['score_percentile'] < self.exit_threshold),
            'exit_long'
        ] = 1

        return dataframe
