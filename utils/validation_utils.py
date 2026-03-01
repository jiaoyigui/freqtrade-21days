# -*- coding: utf-8 -*-
# Source: day20.md - Utility functions
# Freqtrade 21 天从入门到精通

import numpy as np
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime


class BacktestRunner:
    """批量回测运行器"""

    def __init__(self, config_path: str, data_dir: str = "user_data/data"):
        self.config_path = config_path
        self.data_dir = data_dir
        self.results_dir = "user_data/backtest_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_single(self, strategy: str, timerange: str,
                   extra_args: list = None) -> dict:
        """运行单个策略回测"""
        cmd = [
            "freqtrade", "backtesting",
            "--config", self.config_path,
            "--strategy", strategy,
            "--timerange", timerange,
            "--export", "trades",
            "--export-filename",
            f"{self.results_dir}/{strategy}_{timerange}.json"
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            'strategy': strategy,
            'timerange': timerange,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    def run_batch(self, strategies: list, timerange: str) -> list:
        """批量回测多个策略"""
        results = []
        for strategy in strategies:
            print(f"回测：{strategy}...")
            result = self.run_single(strategy, timerange)
            results.append(result)

            if result['returncode'] != 0:
                print(f"  ❌ 失败：{result['stderr'][:200]}")
            else:
                print(f"  ✅ 完成")

        return results


def compute_max_drawdown(returns: np.ndarray) -> float:
    """计算最大回撤"""
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def monte_carlo_permutation_test(strategy_returns: np.ndarray,
                                 n_simulations: int = 10000) -> dict:
    """
    蒙特卡洛置换检验

    H0: 策略收益来自随机
    方法：随机打乱交易顺序，看原始 Sharpe 在随机分布中的位置
    """
    original_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(365)

    random_sharpes = []
    for _ in range(n_simulations):
        shuffled = np.random.permutation(strategy_returns)
        s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(365) if np.std(shuffled) > 0 else 0
        random_sharpes.append(s)

    random_sharpes = np.array(random_sharpes)
    p_value = np.mean(random_sharpes >= original_sharpe)

    return {
        'original_sharpe': original_sharpe,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'percentile': np.mean(random_sharpes < original_sharpe) * 100,
        'random_sharpe_mean': np.mean(random_sharpes),
        'random_sharpe_std': np.std(random_sharpes)
    }


def deflated_sharpe_ratio(observed_sharpe: float,
                          n_trials: int,
                          n_observations: int,
                          skewness: float = 0,
                          kurtosis: float = 3) -> dict:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

    校正多重比较偏差：你试了 100 个策略，最好的那个 Sharpe 可能只是运气
    """
    from scipy.stats import norm

    euler_mascheroni = 0.5772
    expected_max_sharpe = (
        (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials) +
        euler_mascheroni * norm.ppf(1 - (n_trials * np.e))
    )

    sharpe_se = np.sqrt(
        (1 + 0.5 * observed_sharpe**2 -
         skewness * observed_sharpe +
         (kurtosis - 3) / 4 * observed_sharpe**2) /
        (n_observations - 1)
    )

    if sharpe_se > 0:
        dsr = norm.cdf((observed_sharpe - expected_max_sharpe) / sharpe_se)
    else:
        dsr = 0

    return {
        'observed_sharpe': observed_sharpe,
        'expected_max_sharpe': expected_max_sharpe,
        'dsr': dsr,
        'is_significant': dsr > 0.95,
        'n_trials': n_trials,
        'sharpe_haircut': observed_sharpe - expected_max_sharpe
    }
