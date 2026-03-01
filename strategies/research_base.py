# -*- coding: utf-8 -*-
# Source: day20.md - ResearchStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
from abc import abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ResearchStrategy(IStrategy):
    """
    研究框架策略基类

    提供：
    - 统一的指标计算接口
    - 信号质量监控
    - 自动化的信号统计
    """

    INTERFACE_VERSION = 3

    # 子类必须定义
    strategy_family: str = ""
    strategy_version: str = "1.0"

    # 通用风控参数
    max_drawdown_pct: float = 0.15

    def informative_pairs(self):
        """子类可覆盖"""
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_common_indicators(dataframe)
        dataframe = self.add_strategy_indicators(dataframe, metadata)
        return dataframe

    @abstractmethod
    def add_strategy_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """子类实现：添加策略特有指标"""
        pass

    def _add_common_indicators(self, dataframe: DataFrame) -> DataFrame:
        """所有策略共享的基础指标"""
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']

        # ATR
        tr = np.maximum(high - low,
                        np.maximum(abs(high - close.shift(1)),
                                   abs(low - close.shift(1))))
        dataframe['atr_14'] = tr.rolling(14).mean()

        # 波动率
        dataframe['volatility'] = close.pct_change().rolling(20).std()
        dataframe['vol_percentile'] = dataframe['volatility'].rolling(120).rank(pct=True)

        # EMA 族
        for period in [10, 20, 50, 200]:
            dataframe[f'ema_{period}'] = close.ewm(span=period).mean()

        # 成交量
        dataframe['vol_ma_20'] = dataframe['volume'].rolling(20).mean()
        dataframe['vol_ratio'] = dataframe['volume'] / dataframe['vol_ma_20']

        # K 线属性
        range_ = (high - low).replace(0, 1e-8)
        dataframe['body_ratio'] = abs(close - dataframe['open']) / range_
        dataframe['close_position'] = (close - low) / range_

        return dataframe

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                            proposed_stake: float, min_stake, max_stake,
                            leverage: float, entry_tag, side: str,
                            **kwargs) -> float:
        """基于 ATR 的动态仓位"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return proposed_stake

        last_atr = dataframe['atr_14'].iloc[-1]
        last_close = dataframe['close'].iloc[-1]

        if last_atr <= 0 or last_close <= 0:
            return proposed_stake

        risk_per_trade = 0.01
        atr_multiplier = 2.0
        stop_distance = last_atr * atr_multiplier / last_close

        if stop_distance > 0:
            position_size = risk_per_trade / stop_distance
            adjusted_stake = proposed_stake * min(position_size, 1.0)
            return max(min(adjusted_stake, max_stake), min_stake)

        return proposed_stake
