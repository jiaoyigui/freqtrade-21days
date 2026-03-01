# -*- coding: utf-8 -*-
# Source: day15.md - RSRSStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
from sklearn.linear_model import LinearRegression


class RSRSStrategy(IStrategy):
    """
    RSRS 阻力支撑相对强度策略

    原理：
    - 用 High 对 Low 做滚动线性回归，得到斜率 β
    - 标准化 β 并用 R² 修正
    - β 标准化值突破阈值时产生信号

    参考：光大证券《基于阻力支撑相对强度 (RSRS) 的市场择时》
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.08, "60": 0.04, "120": 0.02}
    stoploss = -0.06
    timeframe = '1h'

    # RSRS 参数
    rsrs_window = 18
    rsrs_std_window = 600
    rsrs_buy_threshold = 0.7
    rsrs_sell_threshold = -0.7

    def _calculate_rsrs(self, dataframe: DataFrame) -> DataFrame:
        """计算 RSRS 系列指标"""
        highs = dataframe['high'].values
        lows = dataframe['low'].values
        n = len(dataframe)

        betas = np.full(n, np.nan)
        r_squared = np.full(n, np.nan)

        reg = LinearRegression()

        for i in range(self.rsrs_window, n):
            x = lows[i - self.rsrs_window:i].reshape(-1, 1)
            y = highs[i - self.rsrs_window:i]

            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                continue
            if np.std(x) == 0:
                continue

            reg.fit(x, y)
            betas[i] = reg.coef_[0]
            r_squared[i] = reg.score(x, y)

        dataframe['rsrs_beta'] = betas
        dataframe['rsrs_r2'] = r_squared

        # 标准化
        beta_series = dataframe['rsrs_beta']
        rolling_mean = beta_series.rolling(self.rsrs_std_window, min_periods=60).mean()
        rolling_std = beta_series.rolling(self.rsrs_std_window, min_periods=60).std()

        dataframe['rsrs_std'] = (beta_series - rolling_mean) / rolling_std

        # R² 修正
        dataframe['rsrs_modified'] = dataframe['rsrs_std'] * dataframe['rsrs_r2']

        # 右偏标准化
        dataframe['rsrs_skew'] = dataframe['rsrs_std'] * abs(dataframe['rsrs_std'])

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._calculate_rsrs(dataframe)

        # 辅助指标
        dataframe['ema_20'] = dataframe['close'].ewm(span=20).mean()
        dataframe['ema_60'] = dataframe['close'].ewm(span=60).mean()

        # RSI 过滤
        delta = dataframe['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        dataframe['rsi'] = 100 - (100 / (1 + rs))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsrs_modified'] > self.rsrs_buy_threshold) &
                (dataframe['close'] > dataframe['ema_60']) &
                (dataframe['rsi'] < 70) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'rsrs_bullish')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsrs_modified'] < self.rsrs_sell_threshold) |
                (dataframe['rsi'] > 80)
            ),
            'exit_long'
        ] = 1

        return dataframe
