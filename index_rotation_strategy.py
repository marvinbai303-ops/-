"""指数趋势动量轮动策略（Wind API 版）

策略规则：
1. 计算指数过去 n 天夏普动量：n日涨跌幅 / n日收益波动率。
2. 指数收盘价需同时位于 x 日和 y 日均线上方。
3. 月频调仓，每次最多持有 3 只指数，单只权重上限 33.33%（等权配置）。
4. 调仓信号产生后延迟 T+5 个交易日生效。
5. 若无满足条件指数，则空仓。
6. 输出策略 vs 基准净值曲线，并打印回测报告。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    start_date: str
    end_date: str
    index_pool: List[str]
    benchmark: str
    momentum_window: int = 20
    ma_short: int = 20
    ma_long: int = 60
    rebalance_delay: int = 5
    top_k: int = 3
    annualization: int = 252


class WindDataLoader:
    """Wind API 数据加载器。"""

    def __init__(self) -> None:
        self._w = None

    def _connect(self):
        try:
            from WindPy import w  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "未安装 WindPy。请在 Wind 终端环境安装并配置 WindPy 后重试。"
            ) from exc

        if not w.isconnected():
            w.start()
        if not w.isconnected():
            raise ConnectionError("Wind API 连接失败，请检查 Wind 终端登录状态。")
        self._w = w
        return self._w

    def get_close(self, codes: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
        w = self._connect()
        codes = list(codes)
        raw = w.wsd(codes, "close", start_date, end_date, "PriceAdj=F")
        if raw.ErrorCode != 0:
            raise RuntimeError(f"Wind 数据下载失败，错误码: {raw.ErrorCode}")

        df = pd.DataFrame(raw.Data, index=codes, columns=pd.to_datetime(raw.Times)).T
        df = df.sort_index().ffill().dropna(how="all")
        return df


class IndexTrendMomentumBacktester:
    def __init__(self, config: StrategyConfig, close: pd.DataFrame, benchmark_close: pd.Series):
        self.cfg = config
        self.close = close.copy().sort_index()
        self.benchmark_close = benchmark_close.copy().sort_index()

        aligned_idx = self.close.index.intersection(self.benchmark_close.index)
        self.close = self.close.loc[aligned_idx]
        self.benchmark_close = self.benchmark_close.loc[aligned_idx]

        if self.close.empty:
            raise ValueError("输入价格数据为空，无法回测。")

    def _compute_signals(self) -> pd.DataFrame:
        n = self.cfg.momentum_window
        x = self.cfg.ma_short
        y = self.cfg.ma_long

        daily_ret = self.close.pct_change()
        n_ret = self.close.pct_change(n)
        n_vol = daily_ret.rolling(n).std() * np.sqrt(n)
        sharpe_mom = n_ret / n_vol

        ma_x = self.close.rolling(x).mean()
        ma_y = self.close.rolling(y).mean()
        trend_filter = (self.close > ma_x) & (self.close > ma_y)

        score = sharpe_mom.where(trend_filter)
        return score

    def _monthly_rebalance_dates(self) -> pd.DatetimeIndex:
        # 以每月最后一个交易日作为调仓决策日
        return self.close.resample("ME").last().index

    def _build_target_weights(self, score: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=score.index, columns=score.columns)
        rebalance_dates = self._monthly_rebalance_dates()

        for dt in rebalance_dates:
            if dt not in score.index:
                continue

            row = score.loc[dt].dropna()
            if row.empty:
                continue

            selected = row.sort_values(ascending=False).head(self.cfg.top_k).index
            if len(selected) == 0:
                continue

            w = min(1.0 / len(selected), 1.0 / self.cfg.top_k)
            weights.loc[dt, selected] = w

        return weights

    def run(self) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
        score = self._compute_signals()
        signal_weights = self._build_target_weights(score)

        # T+5 生效：将调仓权重向后移动 5 个交易日
        effective_weights = signal_weights.shift(self.cfg.rebalance_delay)
        effective_weights = effective_weights.replace(0, np.nan).ffill().fillna(0)

        daily_ret = self.close.pct_change().fillna(0)
        strategy_ret = (effective_weights * daily_ret).sum(axis=1)

        benchmark_ret = self.benchmark_close.pct_change().fillna(0)

        nav = pd.DataFrame(index=self.close.index)
        nav["strategy"] = (1 + strategy_ret).cumprod()
        nav["benchmark"] = (1 + benchmark_ret).cumprod()

        report = self._performance_report(strategy_ret, benchmark_ret)
        return nav, report, effective_weights

    def _performance_report(self, strategy_ret: pd.Series, benchmark_ret: pd.Series) -> Dict[str, float]:
        ann = self.cfg.annualization

        def stats(r: pd.Series, prefix: str) -> Dict[str, float]:
            nav = (1 + r).cumprod()
            total_return = nav.iloc[-1] - 1
            ann_return = (1 + total_return) ** (ann / max(len(r), 1)) - 1
            ann_vol = r.std() * np.sqrt(ann)
            sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
            running_max = nav.cummax()
            drawdown = nav / running_max - 1
            max_dd = drawdown.min()
            win_rate = (r > 0).mean()
            calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
            return {
                f"{prefix}_total_return": float(total_return),
                f"{prefix}_annual_return": float(ann_return),
                f"{prefix}_annual_volatility": float(ann_vol),
                f"{prefix}_sharpe": float(sharpe),
                f"{prefix}_max_drawdown": float(max_dd),
                f"{prefix}_win_rate": float(win_rate),
                f"{prefix}_calmar": float(calmar),
            }

        metrics = {}
        metrics.update(stats(strategy_ret, "strategy"))
        metrics.update(stats(benchmark_ret, "benchmark"))
        excess = strategy_ret - benchmark_ret
        metrics.update(stats(excess, "excess"))
        return metrics


def print_report(report: Dict[str, float]) -> None:
    report_series = pd.Series(report)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print("\n===== 回测报告 =====")
        print(report_series)


def plot_nav(nav: pd.DataFrame, save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(nav.index, nav["strategy"], label="Strategy", linewidth=1.8)
    plt.plot(nav.index, nav["benchmark"], label="Benchmark", linewidth=1.5, alpha=0.9)
    plt.title("指数趋势动量策略 vs 基准净值")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"净值图已保存到: {save_path}")
    else:
        plt.show()


def run_with_wind(config: StrategyConfig, save_plot: str = "strategy_vs_benchmark.png") -> None:
    loader = WindDataLoader()
    close = loader.get_close(config.index_pool, config.start_date, config.end_date)
    benchmark_df = loader.get_close([config.benchmark], config.start_date, config.end_date)
    benchmark_close = benchmark_df[config.benchmark]

    backtester = IndexTrendMomentumBacktester(config, close, benchmark_close)
    nav, report, _ = backtester.run()

    plot_nav(nav, save_plot)
    print_report(report)


def run_with_mock_data(save_plot: str = "strategy_vs_benchmark_mock.png") -> None:
    """无 Wind 环境时的示例运行，便于快速验证策略逻辑。"""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", "2024-12-31")

    codes = ["IDX_A", "IDX_B", "IDX_C", "IDX_D", "IDX_E", "IDX_F"]
    ret = np.random.normal(0.0003, 0.012, size=(len(dates), len(codes)))
    close = pd.DataFrame((1 + ret).cumprod(axis=0) * 1000, index=dates, columns=codes)

    bm_ret = np.random.normal(0.0002, 0.01, size=len(dates))
    benchmark = pd.Series((1 + bm_ret).cumprod() * 1000, index=dates, name="BENCH")

    cfg = StrategyConfig(
        start_date="2020-01-01",
        end_date="2024-12-31",
        index_pool=codes,
        benchmark="BENCH",
        momentum_window=20,
        ma_short=20,
        ma_long=60,
        rebalance_delay=5,
        top_k=3,
    )

    backtester = IndexTrendMomentumBacktester(cfg, close, benchmark)
    nav, report, weights = backtester.run()
    plot_nav(nav, save_plot)
    print_report(report)
    print("\n最近5个交易日持仓权重：")
    print(weights.tail())


if __name__ == "__main__":
    # ===== 实盘/历史回测（Wind）示例 =====
    # config = StrategyConfig(
    #     start_date="2018-01-01",
    #     end_date="2024-12-31",
    #     index_pool=["000300.SH", "000905.SH", "399006.SZ", "000852.SH", "399303.SZ"],
    #     benchmark="000300.SH",
    #     momentum_window=20,
    #     ma_short=20,
    #     ma_long=60,
    #     rebalance_delay=5,
    #     top_k=3,
    # )
    # run_with_wind(config, save_plot="strategy_vs_benchmark.png")

    # ===== 本地演示（无需 Wind） =====
    run_with_mock_data()
