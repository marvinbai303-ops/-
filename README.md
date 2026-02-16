# 指数趋势动量轮动策略（Wind API）

该项目实现了一个指数轮动回测脚本：
- 动量因子：`n日涨跌幅 / n日收益波动率`（夏普动量）。
- 趋势过滤：指数需同时在 `x日均线` 与 `y日均线` 上方。
- 调仓规则：月频，调仓信号 `T+5` 执行。
- 持仓规则：每次最多 3 只指数，单只上限 33.33%，无信号则空仓。
- 输出内容：策略 vs 基准净值曲线 + 回测报告。

## 文件
- `index_rotation_strategy.py`：策略主脚本。

## 运行方式

### 1) 无 Wind 环境（示例演示）
```bash
python index_rotation_strategy.py
```
将生成随机模拟数据回测，并输出图像 `strategy_vs_benchmark_mock.png`。

### 2) Wind API 实盘/历史回测
在 `index_rotation_strategy.py` 中取消 `run_with_wind(config)` 示例注释，并按需修改：
- `start_date`, `end_date`
- `index_pool`
- `benchmark`
- `momentum_window`, `ma_short`, `ma_long`

然后运行：
```bash
python index_rotation_strategy.py
```

> 需先安装并可用 WindPy，且 Wind 终端已登录。
