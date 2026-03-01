# -*- coding: utf-8 -*-
# Source: day21.md - 监控告警
# Freqtrade 21天从入门到精通

import requests
from datetime import datetime, timedelta


class TradingMonitor:
    """交易监控系统"""
    
    def __init__(self, instances: dict, webhook_url: str = None):
        """
        Args:
            instances: {name: {'url': 'http://...', 'port': 8081}}
            webhook_url: 告警 webhook（Telegram/Discord/Slack）
        """
        self.instances = instances
        self.webhook_url = webhook_url
    
    def check_all(self) -> dict:
        """检查所有实例状态"""
        status = {}
        for name, config in self.instances.items():
            try:
                url = f"http://{config['url']}:{config['port']}/api/v1/status"
                resp = requests.get(url, timeout=5, 
                                    auth=(config.get('user', ''), config.get('pass', '')))
                data = resp.json()
                status[name] = {
                    'alive': True,
                    'open_trades': len(data) if isinstance(data, list) else 0,
                    'last_check': datetime.now().isoformat()
                }
            except Exception as e:
                status[name] = {
                    'alive': False,
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }
                self._alert(f"🚨 实例 {name} 无响应: {e}")
        
        return status
    
    def check_performance(self, instance_name: str, config: dict) -> dict:
        """检查实例的交易表现"""
        try:
            url = f"http://{config['url']}:{config['port']}/api/v1/profit"
            resp = requests.get(url, timeout=5,
                                auth=(config.get('user', ''), config.get('pass', '')))
            profit = resp.json()
            
            # 告警条件
            if profit.get('profit_all_coin', 0) < -0.1:
                self._alert(f"⚠️ {instance_name} 总亏损超过 10%")
            
            return profit
        except Exception as e:
            return {'error': str(e)}
    
    def check_drawdown(self, daily_returns: list, 
                       max_allowed: float = 0.15) -> bool:
        """实时回撤监控"""
        import numpy as np
        
        cumulative = np.cumprod(1 + np.array(daily_returns))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative[-1] - peak[-1]) / peak[-1]
        
        if abs(drawdown) > max_allowed:
            self._alert(
                f"🔴 回撤警报！当前回撤 {drawdown:.1%}，"
                f"超过阈值 {max_allowed:.1%}。考虑暂停交易。"
            )
            return False
        return True
    
    def _alert(self, message: str):
        """发送告警"""
        print(f"[ALERT] {message}")
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json={'text': message}, timeout=5)
            except Exception:
                pass
