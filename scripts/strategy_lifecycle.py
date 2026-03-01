# -*- coding: utf-8 -*-
# Source: day21.md - 策略生命周期管理
# Freqtrade 21天从入门到精通

import numpy as np
from scipy import stats


def detect_strategy_decay(daily_returns: np.ndarray,
                          window: int = 30,
                          threshold: float = -0.5) -> dict:
    """
    检测策略是否在衰减
    
    方法：
    1. 滚动 Sharpe 的线性趋势
    2. 近期 Sharpe vs 历史 Sharpe
    3. 连续亏损天数
    """
    # 滚动 Sharpe
    rolling_sharpe = []
    for i in range(window, len(daily_returns)):
        chunk = daily_returns[i - window:i]
        s = np.mean(chunk) / np.std(chunk) * np.sqrt(365) if np.std(chunk) > 0 else 0
        rolling_sharpe.append(s)
    
    rolling_sharpe = np.array(rolling_sharpe)
    
    # 趋势检测
    x = np.arange(len(rolling_sharpe))
    slope, _, r_value, p_value, _ = stats.linregress(x, rolling_sharpe)
    
    # 近期 vs 历史
    recent_sharpe = rolling_sharpe[-30:].mean() if len(rolling_sharpe) >= 30 else rolling_sharpe.mean()
    historical_sharpe = rolling_sharpe.mean()
    
    # 连续亏损
    max_losing_streak = 0
    current_streak = 0
    for r in daily_returns[-60:]:
        if r < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0
    
    is_decaying = (
        slope < 0 and p_value < 0.1 and
        recent_sharpe < historical_sharpe * 0.5
    )
    
    return {
        'is_decaying': is_decaying,
        'sharpe_trend_slope': slope,
        'sharpe_trend_p_value': p_value,
        'recent_sharpe': recent_sharpe,
        'historical_sharpe': historical_sharpe,
        'sharpe_ratio': recent_sharpe / historical_sharpe if historical_sharpe != 0 else 0,
        'max_losing_streak': max_losing_streak,
        'recommendation': (
            '🔴 暂停策略，重新评估' if is_decaying else
            '🟡 关注中' if recent_sharpe < historical_sharpe * 0.7 else
            '🟢 正常运行'
        )
    }
