# -*- coding: utf-8 -*-
# Source: day08.md - PairsSpreadStrategy
# Freqtrade 21 天从入门到精通

from freqtrade.strategy import IStrategy
from pandas import DataFrame


class PairsSpreadStrategy(IStrategy):
    """
    配对交易的简化版：交易价差的 Z-score

    方法：计算 ETH/BTC 的价差 Z-score
    当 Z-score 极端时，交易 ETH/USDT（假设 BTC 是"锚"）

    局限：这不是真正的市场中性配对交易，因为没有对冲 BTC 的风险
    真正的配对交易需要合约市场（同时做多做空）
    """
    timeframe = '1h'
    stoploss = -0.05

    def informative_pairs(self):
        return [('BTC/USDT', '1h')]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata['pair'] != 'ETH/USDT':
            return dataframe

        btc_df = self.dp.get_pair_dataframe('BTC/USDT', '1h')

        if len(btc_df) == 0:
            return dataframe

        merged = dataframe.merge(
            btc_df[['date', 'close']].rename(columns={'close': 'btc_close'}),
            on='date', how='left'
        )
        merged['btc_close'] = merged['btc_close'].ffill()

        dataframe['eth_btc_ratio'] = merged['close'] / merged['btc_close']

        lookback = 168
        dataframe['ratio_mean'] = dataframe['eth_btc_ratio'].rolling(lookback).mean()
        dataframe['ratio_std'] = dataframe['eth_btc_ratio'].rolling(lookback).std()
        dataframe['ratio_zscore'] = (
            (dataframe['eth_btc_ratio'] - dataframe['ratio_mean'])
            / dataframe['ratio_std']
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata['pair'] != 'ETH/USDT':
            return dataframe

        dataframe.loc[
            (
                (dataframe['ratio_zscore'] < -2.0) &
                (dataframe['volume'] > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'pair_spread_oversold')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata['pair'] != 'ETH/USDT':
            return dataframe

        dataframe.loc[
            (
                (dataframe['ratio_zscore'] > 0)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'pair_spread_reverted')

        return dataframe
