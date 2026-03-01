# -*- coding: utf-8 -*-
# Source: day10.md - VolatilityBreakoutStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class VolatilityBreakoutStrategy(IStrategy):
    """
    波动率突破策略

    核心逻辑：
    1. 用 GARCH 预测未来波动率
    2. 波动率从低到高转换时（波动率突破）→ 趋势即将启动
    3. 结合价格突破方向决定做多/做空

    学术基础：
    - 波动率聚集意味着低波动率之后大概率还是低波动率
    - 但当波动率突然飙升时，往往伴随着趋势性行情
    - 这就是"波动率压缩→爆发"模式

    参考：Connors & Raschke, "Street Smarts" (1995) - "历史波动率突破"
    """
    timeframe = '4h'
    stoploss = -0.08
    use_custom_stoploss = True

    minimal_roi = {"0": 0.20, "480": 0.10, "960": 0.03}

    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ATR 系列
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_sma_50'] = dataframe['atr'].rolling(50).mean()

        # 波动率比率
        dataframe['vol_ratio'] = dataframe['atr'] / dataframe['atr_sma_50']

        # 布林带宽度
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_width'] = (bb['upperband'] - bb['lowerband']) / bb['middleband']
        dataframe['bb_width_pctile'] = dataframe['bb_width'].rolling(100).rank(pct=True)

        # Keltner 通道
        dataframe['kc_mid'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['kc_upper'] = dataframe['kc_mid'] + dataframe['atr'] * 1.5
        dataframe['kc_lower'] = dataframe['kc_mid'] - dataframe['atr'] * 1.5

        # Squeeze 检测
        dataframe['squeeze'] = (
            (bb['lowerband'] > dataframe['kc_lower']) &
            (bb['upperband'] < dataframe['kc_upper'])
        ).astype(int)

        dataframe['squeeze_release'] = (
            (dataframe['squeeze'] == 0) &
            (dataframe['squeeze'].shift(1) == 1)
        ).astype(int)

        # 动量方向
        dataframe['momentum'] = dataframe['close'] - dataframe['kc_mid']
        dataframe['momentum_increasing'] = (
            dataframe['momentum'] > dataframe['momentum'].shift(1)
        ).astype(int)

        # Donchian 通道
        dataframe['dc_upper_20'] = dataframe['high'].rolling(20).max()
        dataframe['dc_lower_20'] = dataframe['low'].rolling(20).min()

        # ADX 趋势强度
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 策略一：Squeeze 释放 + 向上突破
        dataframe.loc[
            (
                (dataframe['squeeze_release'] == 1) &
                (dataframe['momentum'] > 0) &
                (dataframe['momentum_increasing'] == 1) &
                (dataframe['close'] > dataframe['dc_upper_20'].shift(1)) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'squeeze_breakout_up')

        # 策略二：波动率从极低恢复 + 价格突破
        dataframe.loc[
            (
                (dataframe['bb_width_pctile'] < 0.15) &
                (dataframe['bb_width_pctile'].shift(3) < 0.10) &
                (dataframe['close'] > dataframe['dc_upper_20'].shift(1)) &
                (dataframe['adx'] > 15) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'vol_expansion_breakout')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['momentum'] < 0) &
                (dataframe['momentum'].shift(1) >= 0)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'momentum_reversal')

        return dataframe

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return self.stoploss

        atr = dataframe.iloc[-1].get('atr', 0)
        if atr == 0:
            return self.stoploss

        atr_stop = -(atr * 2.5) / current_rate

        if current_profit > 0.08:
            atr_stop = -(atr * 1.5) / current_rate

        return max(self.stoploss, atr_stop)
