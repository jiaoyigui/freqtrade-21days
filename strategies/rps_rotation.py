# -*- coding: utf-8 -*-
# Source: day15.md - RPSRotationStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas as pd


class RPSRotationStrategy(IStrategy):
    """
    RPS 轮动策略

    原理：
    - 计算所有交易对的 RPS 排名
    - 只交易 RPS 排名前 N 的标的
    - 定期轮换（每周/每两周）
    """

    INTERFACE_VERSION = 3

    minimal_roi = {"0": 0.10, "72": 0.05, "168": 0.02}
    stoploss = -0.08
    timeframe = '4h'

    # RPS 参数
    rps_period = 30
    rps_long_period = 90
    rps_threshold = 80
    rebalance_interval = 42

    def informative_pairs(self):
        """需要所有交易对的数据来计算横截面排名"""
        pairs = self.dp.current_whitelist()
        return [(pair, self.timeframe) for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 区间收益率
        dataframe['return_short'] = dataframe['close'].pct_change(self.rps_period)
        dataframe['return_long'] = dataframe['close'].pct_change(self.rps_long_period)

        # 获取所有交易对的收益率用于排名
        all_returns = {}
        for pair in self.dp.current_whitelist():
            inf_df = self.dp.get_pair_dataframe(pair, self.timeframe)
            if len(inf_df) > 0:
                all_returns[pair] = inf_df.set_index('date')['close'].pct_change(self.rps_period)

        if all_returns:
            returns_df = pd.DataFrame(all_returns)
            # 横截面排名
            ranks = returns_df.rank(axis=1, pct=True) * 100

            pair = metadata['pair']
            if pair in ranks.columns:
                dataframe = dataframe.set_index('date')
                dataframe['rps_rank'] = ranks[pair]
                dataframe = dataframe.reset_index()

        if 'rps_rank' not in dataframe.columns:
            dataframe['rps_rank'] = 50

        # 趋势过滤
        dataframe['ema_50'] = dataframe['close'].ewm(span=50).mean()
        dataframe['ema_200'] = dataframe['close'].ewm(span=200).mean()
        dataframe['uptrend'] = (dataframe['ema_50'] > dataframe['ema_200']).astype(int)

        # 成交量确认
        dataframe['vol_ma'] = dataframe['volume'].rolling(20).mean()
        dataframe['vol_ratio'] = dataframe['volume'] / dataframe['vol_ma']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rps_rank'] > self.rps_threshold) &
                (dataframe['uptrend'] == 1) &
                (dataframe['vol_ratio'] > 0.8) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, f'rps_top_{self.rps_threshold}')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rps_rank'] < 50) |
                (dataframe['uptrend'] == 0)
            ),
            'exit_long'
        ] = 1

        return dataframe
