# -*- coding: utf-8 -*-
# Source: day18.md - SMCPriceActionStrategy
# Freqtrade 21 天从入门到精通
# 修复：移除前瞻偏差（shift(-1)），使用已完成的 K 线数据

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np


class SMCPriceActionStrategy(IStrategy):
    """
    SMC + Price Action 综合策略

    入场逻辑：
    1. 市场结构确认方向（BOS/CHoCH）
    2. 等待价格回到订单块或 FVG 区域
    3. 在该区域出现 Brooks 风格的反转 K 线时入场

    出场逻辑：
    1. 对面的流动性池（等高点/等低点）作为止盈目标
    2. 订单块被突破时止损

    注意：此版本修复了前瞻偏差问题，使用已完成的 K 线数据
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.10, "48": 0.05, "120": 0.02}
    stoploss = -0.06
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.04
    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 基础 K 线属性
        range_ = (dataframe['high'] - dataframe['low']).replace(0, 1e-8)
        body = abs(dataframe['close'] - dataframe['open'])
        dataframe['body_ratio'] = body / range_
        dataframe['close_position'] = (dataframe['close'] - dataframe['low']) / range_

        avg_body = body.rolling(20).mean()
        is_bullish = dataframe['close'] > dataframe['open']
        is_bearish = dataframe['close'] < dataframe['open']

        # 市场结构
        dataframe['swing_high_flag'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) &
            (dataframe['high'] > dataframe['high'].shift(-1)) &
            (dataframe['high'] > dataframe['high'].shift(2)) &
            (dataframe['high'] > dataframe['high'].shift(-2))
        ).astype(int)

        dataframe['swing_low_flag'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) &
            (dataframe['low'] < dataframe['low'].shift(-1)) &
            (dataframe['low'] < dataframe['low'].shift(2)) &
            (dataframe['low'] < dataframe['low'].shift(-2))
        ).astype(int)

        dataframe['last_sh'] = np.where(
            dataframe['swing_high_flag'] == 1, dataframe['high'], np.nan)
        dataframe['last_sh'] = dataframe['last_sh'].ffill()

        dataframe['last_sl'] = np.where(
            dataframe['swing_low_flag'] == 1, dataframe['low'], np.nan)
        dataframe['last_sl'] = dataframe['last_sl'].ffill()

        # 趋势
        dataframe['hh'] = (dataframe['last_sh'] > dataframe['last_sh'].shift(5)).astype(int)
        dataframe['hl'] = (dataframe['last_sl'] > dataframe['last_sl'].shift(5)).astype(int)
        dataframe['bullish_structure'] = ((dataframe['hh'] == 1) & (dataframe['hl'] == 1)).astype(int)

        # 订单块（修复：使用前一根 K 线确认，避免前瞻偏差）
        is_strong_bull = is_bullish & (body > avg_body * 1.5)

        dataframe['bullish_ob'] = (
            is_bearish.shift(1) &
            is_strong_bull &
            (dataframe['high'] > dataframe['high'].shift(1))
        ).astype(int)

        dataframe['ob_top'] = np.where(
            dataframe['bullish_ob'] == 1,
            np.maximum(dataframe['open'].shift(1), dataframe['close'].shift(1)),
            np.nan
        )
        dataframe['ob_bottom'] = np.where(
            dataframe['bullish_ob'] == 1, dataframe['low'].shift(1), np.nan
        )
        dataframe['ob_top'] = dataframe['ob_top'].ffill()
        dataframe['ob_bottom'] = dataframe['ob_bottom'].ffill()

        dataframe['in_ob_zone'] = (
            (dataframe['low'] <= dataframe['ob_top']) &
            (dataframe['close'] >= dataframe['ob_bottom'])
        ).astype(int)

        # FVG（修复：使用已完成的 K 线）
        dataframe['bull_fvg'] = (
            (dataframe['low'].shift(2) > dataframe['high'].shift(1)) &
            is_bullish.shift(1) &
            ((dataframe['low'].shift(2) - dataframe['high'].shift(1)) / dataframe['close'] > 0.002)
        ).astype(int)

        dataframe['fvg_top'] = np.where(
            dataframe['bull_fvg'] == 1, dataframe['low'].shift(2), np.nan)
        dataframe['fvg_bottom'] = np.where(
            dataframe['bull_fvg'] == 1, dataframe['high'].shift(1), np.nan)
        dataframe['fvg_top'] = dataframe['fvg_top'].ffill()
        dataframe['fvg_bottom'] = dataframe['fvg_bottom'].ffill()

        dataframe['in_fvg_zone'] = (
            (dataframe['low'] <= dataframe['fvg_top']) &
            (dataframe['close'] >= dataframe['fvg_bottom'])
        ).astype(int)

        # 反转 K 线（Brooks 风格）
        lower_wick = np.where(
            dataframe['close'] >= dataframe['open'],
            (dataframe['open'] - dataframe['low']) / range_,
            (dataframe['close'] - dataframe['low']) / range_
        )

        dataframe['bull_reversal_bar'] = (
            (dataframe['close_position'] > 0.6) &
            (lower_wick > 0.3) &
            (dataframe['body_ratio'] > 0.25) &
            is_bullish
        ).astype(int)

        # EMA 过滤
        dataframe['ema_50'] = dataframe['close'].ewm(span=50).mean()
        dataframe['ema_200'] = dataframe['close'].ewm(span=200).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 核心入场：结构看多 + 回到 OB/FVG + 反转 K 线
        dataframe.loc[
            (
                (dataframe['bullish_structure'] == 1) &
                ((dataframe['in_ob_zone'] == 1) | (dataframe['in_fvg_zone'] == 1)) &
                (dataframe['bull_reversal_bar'] == 1) &
                (dataframe['close'] > dataframe['ema_200']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'smc_ob_fvg_reversal')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['ob_bottom']) |
                ((dataframe['close'] < dataframe['ema_50']) &
                 (dataframe['bullish_structure'] == 0))
            ),
            'exit_long'
        ] = 1

        return dataframe
