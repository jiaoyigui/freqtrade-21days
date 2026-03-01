# -*- coding: utf-8 -*-
# Source: day11.md - MultiFactorStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import pandas as pd
import talib.abstract as ta


class MultiFactorStrategy(IStrategy):
    """
    多因子选币策略

    流程：
    1. 对所有候选币计算因子值
    2. 综合排名，选出 Top N
    3. 只交易 Top N 的币

    因子权重可以通过 Hyperopt 优化
    """
    timeframe = '4h'
    stoploss = -0.08

    # 因子权重（可优化）
    w_momentum = DecimalParameter(0.0, 1.0, default=0.4, decimals=1, space='buy')
    w_volatility = DecimalParameter(0.0, 1.0, default=0.3, decimals=1, space='buy')
    w_volume = DecimalParameter(0.0, 1.0, default=0.3, decimals=1, space='buy')

    top_n = IntParameter(3, 10, default=5, space='buy')

    minimal_roi = {"0": 0.10, "360": 0.05, "720": 0.02}

    def bot_loop_start(self, current_time, **kwargs):
        """每轮循环计算所有币的因子排名"""
        pairs = self.dp.current_whitelist()
        factor_scores = {}

        for pair in pairs:
            df = self.dp.get_pair_dataframe(pair, self.timeframe)
            if len(df) < 200:
                continue

            close = df['close']
            volume = df['volume']

            # 动量因子
            momentum = (close.iloc[-1] / close.iloc[-31]) - 1 if len(close) > 31 else 0

            # 波动率因子
            returns = close.pct_change().dropna()
            vol = returns.iloc[-30:].std() if len(returns) > 30 else 999
            vol_score = -vol

            # 成交量因子
            vol_ratio = (volume.iloc[-10:].mean() / volume.iloc[-50:].mean()
                        if len(volume) > 50 else 1)

            factor_scores[pair] = {
                'momentum': momentum,
                'volatility': vol_score,
                'volume': vol_ratio
            }

        if not factor_scores:
            self.top_pairs = []
            return

        # 横截面排名
        score_df = pd.DataFrame(factor_scores).T
        for col in score_df.columns:
            score_df[f'{col}_rank'] = score_df[col].rank(pct=True)

        # 综合得分
        weights = {
            'momentum_rank': self.w_momentum.value,
            'volatility_rank': self.w_volatility.value,
            'volume_rank': self.w_volume.value
        }
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1

        score_df['composite'] = sum(
            score_df[col] * w / total_weight
            for col, w in weights.items()
        )

        # 选出 Top N
        self.top_pairs = score_df.nlargest(self.top_n.value, 'composite').index.tolist()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_top = metadata['pair'] in getattr(self, 'top_pairs', [])

        dataframe.loc[
            (
                is_top &
                (dataframe['ema_20'] > dataframe['ema_50']) &
                (dataframe['adx'] > 20) &
                (dataframe['rsi'] < 65) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'multifactor_top')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_top = metadata['pair'] in getattr(self, 'top_pairs', [])

        dataframe.loc[
            (
                (~is_top) |
                (dataframe['rsi'] > 75) |
                (
                    (dataframe['ema_20'] < dataframe['ema_50']) &
                    (dataframe['ema_20'].shift(1) >= dataframe['ema_50'].shift(1))
                )
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'factor_rotation')

        return dataframe
