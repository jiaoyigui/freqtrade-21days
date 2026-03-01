# -*- coding: utf-8 -*-
# Source: day15.md - RSRS_RPS_Strategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression


class RSRS_RPS_Strategy(IStrategy):
    """
    RSRS 择时 + RPS 选币

    逻辑：
    1. RSRS > 阈值 → 市场环境适合做多
    2. RPS > 80 → 只做最强的标的
    3. 两个条件同时满足才开仓
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.10, "48": 0.05, "120": 0.02}
    stoploss = -0.07
    timeframe = '4h'

    def _calculate_rsrs(self, dataframe: DataFrame) -> DataFrame:
        """RSRS 计算"""
        highs = dataframe['high'].values
        lows = dataframe['low'].values
        n = len(dataframe)
        window = 18

        betas = np.full(n, np.nan)
        r2s = np.full(n, np.nan)
        reg = LinearRegression()

        for i in range(window, n):
            x = lows[i-window:i].reshape(-1, 1)
            y = highs[i-window:i]
            if np.std(x) == 0 or np.any(np.isnan(x)) or np.any(np.isnan(y)):
                continue
            reg.fit(x, y)
            betas[i] = reg.coef_[0]
            r2s[i] = reg.score(x, y)

        dataframe['rsrs_beta'] = betas
        beta_s = dataframe['rsrs_beta']
        mu = beta_s.rolling(300, min_periods=60).mean()
        sigma = beta_s.rolling(300, min_periods=60).std()
        dataframe['rsrs_std'] = (beta_s - mu) / sigma
        dataframe['rsrs_modified'] = dataframe['rsrs_std'] * r2s

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._calculate_rsrs(dataframe)

        # RPS（简化版：用自身动量近似）
        dataframe['momentum_20'] = dataframe['close'].pct_change(20)
        dataframe['momentum_60'] = dataframe['close'].pct_change(60)
        dataframe['rps_proxy'] = 0.6 * dataframe['momentum_20'] + 0.4 * dataframe['momentum_60']

        # 波动率过滤
        dataframe['volatility'] = dataframe['close'].pct_change().rolling(20).std()
        dataframe['vol_percentile'] = dataframe['volatility'].rolling(120).rank(pct=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsrs_modified'] > 0.7) &
                (dataframe['rps_proxy'] > dataframe['rps_proxy'].rolling(60).quantile(0.8)) &
                (dataframe['vol_percentile'] < 0.8) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'rsrs_rps_combo')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsrs_modified'] < -0.7) |
                (dataframe['rps_proxy'] < dataframe['rps_proxy'].rolling(60).quantile(0.2))
            ),
            'exit_long'
        ] = 1

        return dataframe
