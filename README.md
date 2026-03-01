# Freqtrade 21天从入门到精通 — 配套代码

本目录包含教程中所有可运行的策略代码和工具函数。

## 目录结构

```
code/
├── strategies/          # 策略文件（可直接放入 user_data/strategies/）
├── utils/               # 工具函数库
├── configs/             # 配置文件模板
└── scripts/             # 运维脚本
```

## 策略索引

### 经典量化策略

| 文件 | 策略名 | 来源 | 类型 |
|------|--------|------|------|
| `bollinger_mean_revert.py` | BollingerMeanRevert | Ch08 | 均值回归 |
| `pairs_spread.py` | PairsSpreadStrategy | Ch08 | 配对交易 |
| `modern_turtle.py` | ModernTurtleStrategy | Ch09 | 趋势跟随 |
| `dual_momentum.py` | DualMomentumStrategy | Ch09 | 趋势跟随 |
| `volatility_breakout.py` | VolatilityBreakoutStrategy | Ch10 | 波动率突破 |
| `volatility_sell.py` | VolatilitySellStrategy | Ch10 | 卖波动率 |
| `multi_factor.py` | MultiFactorStrategy | Ch11 | 多因子 |
| `freqai_robust.py` | FreqAIRobustStrategy | Ch12 | 机器学习 |
| `meta_strategy.py` | MetaStrategy | Ch14 | 策略组合 |

### 价格行为与微观结构

| 文件 | 策略名 | 来源 | 类型 |
|------|--------|------|------|
| `smc_basic.py` | SMCStrategy | Ch13 | SMC 入门 |
| `funding_rate.py` | FundingRateStrategy | Ch13 | 资金费率套利 |
| `brooks_pa_full.py` | BrooksPriceActionStrategy | Ch17 | Al Brooks PA |
| `smc_pa.py` | SMCPriceActionStrategy | Ch18 | SMC + PA |
| `chan_pa.py` | ChanPriceActionStrategy | Ch19 | 缠论 + PA |

### 中国量化经典

| 文件 | 策略名 | 来源 | 类型 |
|------|--------|------|------|
| `rsrs.py` | RSRSStrategy | Ch15 | RSRS 择时 |
| `rps_rotation.py` | RPSRotationStrategy | Ch15 | RPS 轮动 |
| `rsrs_rps_combined.py` | RSRS_RPS_Strategy | Ch15 | RSRS+RPS 联合 |
| `alpha101.py` | Alpha101Strategy | Ch16 | Alpha 101 因子 |

### 研究框架

| 文件 | 策略名 | 来源 | 类型 |
|------|--------|------|------|
| `research_base.py` | ResearchStrategy | Ch20 | 策略基类 |
| `optimizable.py` | OptimizableStrategy | Ch20 | Hyperopt 模板 |

## 工具函数

| 文件 | 功能 | 来源 |
|------|------|------|
| `data_quality.py` | 数据质量检查与清洗 | Ch04 |
| `backtest_utils.py` | 回测偏差检测 | Ch05 |
| `indicator_utils.py` | 自适应指标与信号处理 | Ch06 |
| `risk_utils.py` | 凯利公式、破产概率、VaR/CVaR | Ch07 |
| `mean_revert_utils.py` | OU 过程估计、ADF 检验、协整 | Ch08 |
| `factor_utils.py` | 因子构建、正交化、拥挤检测 | Ch11 |
| `rsrs_rps_utils.py` | RSRS/RPS 计算、IC 分析 | Ch15 |
| `alpha_operators.py` | Alpha 101 基础算子库 | Ch16 |
| `validation_utils.py` | 蒙特卡洛检验、DSR、Walk-Forward | Ch20 |

## 快速开始

```bash
# 1. 复制策略到 Freqtrade
cp code/strategies/*.py /path/to/freqtrade/user_data/strategies/

# 2. 复制工具函数
cp -r code/utils/ /path/to/freqtrade/user_data/strategies/utils/

# 3. 回测示例
freqtrade backtesting --strategy BollingerMeanRevert --timeframe 1h --timerange 20230101-20240101
```

## 风险提示

⚠️ 所有策略仅供学习研究，不构成投资建议。请在 dry-run 模式下充分测试后再考虑实盘。
