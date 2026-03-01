# -*- coding: utf-8 -*-
# Source: day17.md - BrooksPriceActionStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np


class BrooksPriceActionStrategy(IStrategy):
    """
    Al Brooks 价格行为策略

    核心逻辑：
    1. 判断市场状态（趋势/区间/反转）
    2. 趋势中：等待回调，用二次入场做多/做空
    3. 区间中：在边界做反向交易
    4. 反转时：识别楔形和强反转 K 线

    参考：Al Brooks "Trading Price Action" 三部曲
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.08, "30": 0.04, "60": 0.02}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # K 线属性
        range_ = dataframe['high'] - dataframe['low']
        range_ = range_.replace(0, 1e-8)
        body = abs(dataframe['close'] - dataframe['open'])

        dataframe['body_ratio'] = body / range_
        dataframe['close_position'] = (dataframe['close'] - dataframe['low']) / range_

        dataframe['upper_wick'] = np.where(
            dataframe['close'] >= dataframe['open'],
            (dataframe['high'] - dataframe['close']) / range_,
            (dataframe['high'] - dataframe['open']) / range_
        )
        dataframe['lower_wick'] = np.where(
            dataframe['close'] >= dataframe['open'],
            (dataframe['open'] - dataframe['low']) / range_,
            (dataframe['close'] - dataframe['low']) / range_
        )

        # 市场状态判断
        dataframe['ema_20'] = dataframe['close'].ewm(span=20).mean()
        dataframe['ema_slope'] = dataframe['ema_20'].diff(5) / dataframe['ema_20'].shift(5)

        # 连续同向 K 线计数
        dataframe['bull_bar'] = (dataframe['close'] > dataframe['open']).astype(int)
        dataframe['bear_bar'] = (dataframe['close'] < dataframe['open']).astype(int)

        dataframe['consec_bull'] = dataframe['bull_bar'].groupby(
            (dataframe['bull_bar'] != dataframe['bull_bar'].shift()).cumsum()
        ).cumsum()
        dataframe['consec_bear'] = dataframe['bear_bar'].groupby(
            (dataframe['bear_bar'] != dataframe['bear_bar'].shift()).cumsum()
        ).cumsum()

        dataframe['trend_strength'] = dataframe['ema_slope'].rolling(10).mean()

        # 信号 K 线
        dataframe['bull_signal'] = (
            (dataframe['close_position'] > 0.6) &
            (dataframe['lower_wick'] > 0.3) &
            (dataframe['body_ratio'] > 0.25) &
            (dataframe['close'] > dataframe['open'])
        ).astype(int)

        dataframe['bear_signal'] = (
            (dataframe['close_position'] < 0.4) &
            (dataframe['upper_wick'] > 0.3) &
            (dataframe['body_ratio'] > 0.25) &
            (dataframe['close'] < dataframe['open'])
        ).astype(int)

        # 二次入场检测
        dataframe['recent_bull_signals'] = dataframe['bull_signal'].rolling(10).sum()
        dataframe['second_entry_bull'] = (
            (dataframe['bull_signal'] == 1) &
            (dataframe['recent_bull_signals'] >= 2)
        ).astype(int)

        dataframe['recent_bear_signals'] = dataframe['bear_signal'].rolling(10).sum()
        dataframe['second_entry_bear'] = (
            (dataframe['bear_signal'] == 1) &
            (dataframe['recent_bear_signals'] >= 2)
        ).astype(int)

        # 楔形检测（简化）
        dataframe['is_swing_low'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) &
            (dataframe['low'] < dataframe['low'].shift(-1)) &
            (dataframe['low'] < dataframe['low'].shift(2)) &
            (dataframe['low'] < dataframe['low'].shift(-2))
        ).astype(int)

        dataframe['swing_low_count'] = dataframe['is_swing_low'].rolling(30).sum()
        dataframe['price_declining'] = (dataframe['low'] < dataframe['low'].rolling(30).mean()).astype(int)
        dataframe['wedge_bull'] = (
            (dataframe['swing_low_count'] >= 3) &
            (dataframe['price_declining'] == 1) &
            (dataframe['bull_signal'] == 1)
        ).astype(int)

        # 支撑阻力
        dataframe['resistance'] = dataframe['high'].rolling(20).max()
        dataframe['support'] = dataframe['low'].rolling(20).min()
        dataframe['range_width'] = (dataframe['resistance'] - dataframe['support']) / dataframe['close']

        dataframe['range_position'] = (
            (dataframe['close'] - dataframe['support']) /
            (dataframe['resistance'] - dataframe['support'] + 1e-8)
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 策略 1：趋势中的二次入场
        dataframe.loc[
            (
                (dataframe['second_entry_bull'] == 1) &
                (dataframe['trend_strength'] > 0.001) &
                (dataframe['close'] > dataframe['ema_20']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'brooks_2nd_entry')

        # 策略 2：楔形反转
        dataframe.loc[
            (
                (dataframe['wedge_bull'] == 1) &
                (dataframe['range_position'] < 0.3) &
                (dataframe['volume'] > 0) &
                (dataframe['enter_long'] != 1)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'brooks_wedge')

        # 策略 3：区间底部反弹
        dataframe.loc[
            (
                (dataframe['range_position'] < 0.15) &
                (dataframe['bull_signal'] == 1) &
                (dataframe['range_width'] > 0.03) &
                (dataframe['volume'] > 0) &
                (dataframe['enter_long'] != 1)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'brooks_range_bottom')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['bear_signal'] == 1) &
                (dataframe['range_position'] > 0.85)
            ) |
            (
                (dataframe['consec_bear'] >= 3) &
                (dataframe['trend_strength'] < -0.002)
            ),
            'exit_long'
        ] = 1

        return dataframe
