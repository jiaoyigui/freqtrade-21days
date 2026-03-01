# -*- coding: utf-8 -*-
# Source: day09.md - ModernTurtleStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta


class ModernTurtleStrategy(IStrategy):
    """
    海龟交易法的现代演绎

    改进点：
    1. 用 ATR 标准化的 Donchian 通道替代固定天数
    2. 加入波动率过滤（低波动率时不交易）
    3. 加入趋势强度确认（ADX）
    4. 动态仓位管理（基于波动率的风险平价）

    参考：
    - Curtis Faith, "Way of the Turtle" (2007)
    - Covel, "Trend Following" (2004)
    """
    timeframe = '4h'
    stoploss = -0.15
    use_custom_stoploss = True

    # Donchian 通道参数
    entry_period = IntParameter(15, 30, default=20, space='buy')
    exit_period = IntParameter(8, 15, default=10, space='sell')
    atr_period = IntParameter(10, 25, default=20, space='buy')
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space='stoploss')

    minimal_roi = {"0": 0.30, "480": 0.15, "1440": 0.05}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Donchian 通道
        dataframe['dc_upper'] = dataframe['high'].rolling(self.entry_period.value).max()
        dataframe['dc_lower'] = dataframe['low'].rolling(self.entry_period.value).min()
        dataframe['dc_mid'] = (dataframe['dc_upper'] + dataframe['dc_lower']) / 2

        # 退出通道
        dataframe['dc_exit_lower'] = dataframe['low'].rolling(self.exit_period.value).min()

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']
        dataframe['atr_sma'] = dataframe['atr'].rolling(50).mean()

        # 趋势强度
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # 趋势方向确认
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # 动量确认
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['dc_upper'].shift(1)) &
                (dataframe['adx'] > 20) &
                (dataframe['ema_50'] > dataframe['ema_200']) &
                (dataframe['roc'] > 0) &
                (dataframe['atr'] > dataframe['atr_sma'] * 0.8) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'turtle_breakout')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['dc_exit_lower'].shift(1))
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'turtle_exit')

        return dataframe

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return self.stoploss

        atr = dataframe.iloc[-1].get('atr', 0)
        if atr == 0:
            return self.stoploss

        atr_stop = -(atr * self.atr_multiplier.value) / current_rate

        if current_profit > 0.10:
            atr_stop = -(atr * 1.5) / current_rate
        if current_profit > 0.20:
            atr_stop = -(atr * 1.0) / current_rate

        return max(self.stoploss, atr_stop)

    def custom_stake_amount(self, pair, current_time, current_rate,
                            proposed_stake, min_stake, max_stake,
                            leverage, entry_tag, side, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return proposed_stake

        atr = dataframe.iloc[-1].get('atr', 0)
        if atr == 0:
            return proposed_stake

        total_capital = self.wallets.get_total_stake_amount()
        risk_per_trade = 0.02
        risk_amount = total_capital * risk_per_trade

        stop_distance = atr * self.atr_multiplier.value
        position_size = risk_amount / (stop_distance / current_rate)

        return max(min_stake, min(position_size, max_stake))
