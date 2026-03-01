# -*- coding: utf-8 -*-
# Source: day13.md - SMCStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import numpy as np
import talib.abstract as ta


class SMCStrategy(IStrategy):
    """
    Smart Money Concepts 策略

    核心逻辑：
    1. 识别市场结构（BOS / CHoCH）
    2. 标记 Order Block 和 FVG
    3. 等待价格回到 Order Block / FVG 区域
    4. 在流动性扫荡后反转入场
    """
    timeframe = '1h'
    stoploss = -0.04
    use_custom_stoploss = True

    minimal_roi = {"0": 0.06, "180": 0.03, "480": 0.01}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Swing High / Swing Low 识别
        lookback = 5
        dataframe['swing_high'] = dataframe['high'].rolling(
            lookback * 2 + 1, center=True
        ).apply(lambda x: x[lookback] if x[lookback] == x.max() else np.nan)

        dataframe['swing_low'] = dataframe['low'].rolling(
            lookback * 2 + 1, center=True
        ).apply(lambda x: x[lookback] if x[lookback] == x.min() else np.nan)

        dataframe['prev_swing_high'] = dataframe['swing_high'].ffill()
        dataframe['prev_swing_low'] = dataframe['swing_low'].ffill()

        # Break of Structure (BOS)
        dataframe['bull_bos'] = (
            (dataframe['close'] > dataframe['prev_swing_high'].shift(1)) &
            (dataframe['close'].shift(1) <= dataframe['prev_swing_high'].shift(1))
        ).astype(int)

        dataframe['bear_bos'] = (
            (dataframe['close'] < dataframe['prev_swing_low'].shift(1)) &
            (dataframe['close'].shift(1) >= dataframe['prev_swing_low'].shift(1))
        ).astype(int)

        # Fair Value Gap (FVG)
        dataframe['bull_fvg'] = (
            dataframe['low'] > dataframe['high'].shift(2)
        ).astype(int)

        dataframe['fvg_top'] = np.where(
            dataframe['bull_fvg'] == 1,
            dataframe['low'],
            np.nan
        )
        dataframe['fvg_bottom'] = np.where(
            dataframe['bull_fvg'] == 1,
            dataframe['high'].shift(2),
            np.nan
        )
        dataframe['fvg_top'] = dataframe['fvg_top'].ffill()
        dataframe['fvg_bottom'] = dataframe['fvg_bottom'].ffill()

        dataframe['fvg_filled'] = (
            dataframe['low'] <= dataframe['fvg_bottom']
        ).astype(int)

        # Order Block 识别
        dataframe['bull_ob_high'] = np.nan
        dataframe['bull_ob_low'] = np.nan

        for i in range(len(dataframe)):
            if dataframe['bull_bos'].iloc[i] == 1:
                for j in range(i-1, max(i-10, 0), -1):
                    if dataframe['close'].iloc[j] < dataframe['open'].iloc[j]:
                        dataframe.loc[dataframe.index[i], 'bull_ob_high'] = dataframe['high'].iloc[j]
                        dataframe.loc[dataframe.index[i], 'bull_ob_low'] = dataframe['low'].iloc[j]
                        break

        dataframe['bull_ob_high'] = dataframe['bull_ob_high'].ffill()
        dataframe['bull_ob_low'] = dataframe['bull_ob_low'].ffill()

        # 辅助指标（先定义再使用）
        dataframe['is_bull'] = (dataframe['close'] > dataframe['open']).astype(int)

        # Liquidity Sweep 检测
        dataframe['sweep_low'] = (
            (dataframe['low'] < dataframe['prev_swing_low'].shift(1)) &
            (dataframe['close'] > dataframe['prev_swing_low'].shift(1)) &
            (dataframe['is_bull'] == 1)
        ).astype(int)

        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 入场一：价格回到 Order Block 区域 + 反转确认
        dataframe.loc[
            (
                (dataframe['low'] <= dataframe['bull_ob_high']) &
                (dataframe['close'] >= dataframe['bull_ob_low']) &
                (dataframe['is_bull'] == 1) &
                (dataframe['body_ratio'] > 0.4) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'smc_order_block')

        # 入场二：流动性扫荡后反转
        dataframe.loc[
            (
                (dataframe['sweep_low'] == 1) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['rsi'] < 45) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'smc_liquidity_sweep')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['bear_bos'] == 1) |
                (dataframe['rsi'] > 75)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'smc_structure_break')

        return dataframe
