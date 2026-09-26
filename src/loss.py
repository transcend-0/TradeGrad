import json
import logging
import math
import sys
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .job import run_job

logger = logging.getLogger(__name__)


class TextLoss(ABC):
    def __init__(self, llm):
        self.llm = llm

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Run the backtest and record `strategy.performance`."""
        pass


class StrategyReward(TextLoss):
    """One continuous backtest of a cross-sectional strategy over [start_time, end_time]."""

    def __init__(
        self,
        llm,
        evaluate_script: str,
        date2symbol_dir: str,
        data_dir: str,
        start_time: str,
        end_time: str,
    ):
        super().__init__(llm)
        self.evaluate_script = evaluate_script
        self.date2symbol_dir = date2symbol_dir
        self.data_dir = data_dir
        self.start_time = start_time
        self.end_time = end_time

    def forward(self, strategy):
        cmd = [
            sys.executable, self.evaluate_script,
            "--strategy_path", str(strategy.strategy_path),
            "--result_dir", str(strategy.strategy_dir),
            "--date2symbol_dir", self.date2symbol_dir,
            "--data_dir", self.data_dir,
            "--start_time", self.start_time,
            "--end_time", self.end_time,
        ]
        run_time, error = run_job(strategy.strategy_dir, cmd)

        if error:
            performance = {"error": error}
        else:
            with open(strategy.strategy_dir / "metrics.json", "r") as f:
                performance = json.load(f)
            performance["run_time"] = run_time
            with open(strategy.strategy_dir / "metrics.json", "w") as f:
                json.dump(performance, f, indent=4)

        strategy.performance = performance
        logger.info(f"Strategy {strategy.id} performance: {performance}")
        return performance


def split_time_windows_by_year(start_time: str, end_time: str):
    """Split [start_time, end_time] (both "YYYY-MM-DD") into one window per calendar year."""
    start_date = datetime.strptime(start_time, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_time, "%Y-%m-%d").date()
    windows = []
    for year in range(start_date.year, end_date.year + 1):
        window_start = max(start_date, date(year, 1, 1))
        window_end = min(end_date, date(year, 12, 31))
        windows.append((window_start.isoformat(), window_end.isoformat()))
    return windows


def run_window_backtest(strategy_path, window_dir, evaluate_script, date2symbol_dir, data_dir, start_time, end_time):
    window_dir = Path(window_dir)
    cmd = [
        sys.executable, evaluate_script,
        "--strategy_path", str(strategy_path),
        "--result_dir", str(window_dir),
        "--date2symbol_dir", date2symbol_dir,
        "--data_dir", data_dir,
        "--start_time", start_time,
        "--end_time", end_time,
    ]
    run_time, error = run_job(window_dir, cmd)

    if error:
        performance = {"error": error}
    else:
        with open(window_dir / "metrics.json", "r") as f:
            performance = json.load(f)
        performance["run_time"] = run_time
        with open(window_dir / "metrics.json", "w") as f:
            json.dump(performance, f, indent=4)

    performance["start_time"] = start_time
    performance["end_time"] = end_time
    return performance


class WindowStrategyReward(TextLoss):
    """
    Splits [start_time, end_time] into yearly windows, backtests each window independently, and
    scores the strategy by the average `combined_score` of its worst-performing windows
    (tail-risk-aware reward). This is the training objective TradeTGD is run under.
    """

    def __init__(
        self,
        llm,
        evaluate_script: str,
        date2symbol_dir: str,
        data_dir: str,
        start_time: str,
        end_time: str,
        worst_ratio: float = 0.3,
        max_workers: int = 4,
    ):
        super().__init__(llm)
        self.evaluate_script = evaluate_script
        self.date2symbol_dir = date2symbol_dir
        self.data_dir = data_dir
        self.start_time = start_time
        self.end_time = end_time
        self.worst_ratio = worst_ratio
        self.max_workers = max_workers
        self.windows = split_time_windows_by_year(start_time, end_time)

    def forward(self, strategy):
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.windows))) as executor:
            futures = {
                executor.submit(
                    run_window_backtest,
                    strategy.strategy_path,
                    strategy.strategy_dir / f"window_{w_start}_{w_end}",
                    self.evaluate_script,
                    self.date2symbol_dir,
                    self.data_dir,
                    w_start,
                    w_end,
                ): i
                for i, (w_start, w_end) in enumerate(self.windows)
            }
            window_performances = [None] * len(self.windows)
            for future in as_completed(futures):
                window_performances[futures[future]] = future.result()

        scores = [p.get("combined_score", 0) for p in window_performances]
        n_worst = max(1, math.ceil(len(scores) * self.worst_ratio))
        worst_avg_score = round(sum(sorted(scores)[:n_worst]) / n_worst, 2)

        performance = {
            "combined_score": worst_avg_score,
            "windows": window_performances,
        }
        errors = [p["error"] for p in window_performances if p.get("error")]
        if errors:
            performance["error"] = "\n\n".join(errors)

        strategy.performance = performance
        with open(strategy.strategy_dir / "metrics.json", "w") as f:
            json.dump(performance, f, indent=4)

        logger.info(f"Strategy {strategy.id} window performance: {performance}")
        return performance


class TimeStrategyReward(TextLoss):
    """One continuous backtest of a single-symbol timing strategy over [start_time, end_time]."""

    def __init__(
        self,
        llm,
        evaluate_script: str,
        data_path: str,
        start_time: str,
        end_time: str,
        symbol: str = "target_symbol",
    ):
        super().__init__(llm)
        self.evaluate_script = evaluate_script
        self.data_path = data_path
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time

    def forward(self, strategy):
        cmd = [
            sys.executable, self.evaluate_script,
            "--strategy_path", str(strategy.strategy_path),
            "--result_dir", str(strategy.strategy_dir),
            "--data_path", self.data_path,
            "--symbol", self.symbol,
            "--start_time", self.start_time,
            "--end_time", self.end_time,
        ]
        run_time, error = run_job(strategy.strategy_dir, cmd)

        if error:
            performance = {"error": error}
        else:
            with open(strategy.strategy_dir / "metrics.json", "r") as f:
                performance = json.load(f)
            performance["run_time"] = run_time
            with open(strategy.strategy_dir / "metrics.json", "w") as f:
                json.dump(performance, f, indent=4)

        strategy.performance = performance
        logger.info(f"Strategy {strategy.id} performance: {performance}")
        return performance


def run_time_window_backtest(strategy_path, window_dir, evaluate_script, data_path, symbol, start_time, end_time):
    window_dir = Path(window_dir)
    cmd = [
        sys.executable, evaluate_script,
        "--strategy_path", str(strategy_path),
        "--result_dir", str(window_dir),
        "--data_path", data_path,
        "--symbol", symbol,
        "--start_time", start_time,
        "--end_time", end_time,
    ]
    run_time, error = run_job(window_dir, cmd)

    if error:
        performance = {"error": error}
    else:
        with open(window_dir / "metrics.json", "r") as f:
            performance = json.load(f)
        performance["run_time"] = run_time
        with open(window_dir / "metrics.json", "w") as f:
            json.dump(performance, f, indent=4)

    performance["start_time"] = start_time
    performance["end_time"] = end_time
    return performance


class WindowTimeStrategyReward(TextLoss):
    """Time-series analog of `WindowStrategyReward`: yearly windows, worst-30% CVaR aggregation."""

    def __init__(
        self,
        llm,
        evaluate_script: str,
        data_path: str,
        start_time: str,
        end_time: str,
        symbol: str = "target_symbol",
        worst_ratio: float = 0.3,
        max_workers: int = 4,
    ):
        super().__init__(llm)
        self.evaluate_script = evaluate_script
        self.data_path = data_path
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.worst_ratio = worst_ratio
        self.max_workers = max_workers
        self.windows = split_time_windows_by_year(start_time, end_time)

    def forward(self, strategy):
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(self.windows))) as executor:
            futures = {
                executor.submit(
                    run_time_window_backtest,
                    strategy.strategy_path,
                    strategy.strategy_dir / f"window_{w_start}_{w_end}",
                    self.evaluate_script,
                    self.data_path,
                    self.symbol,
                    w_start,
                    w_end,
                ): i
                for i, (w_start, w_end) in enumerate(self.windows)
            }
            window_performances = [None] * len(self.windows)
            for future in as_completed(futures):
                window_performances[futures[future]] = future.result()

        scores = [p.get("combined_score", 0) for p in window_performances]
        n_worst = max(1, math.ceil(len(scores) * self.worst_ratio))
        worst_avg_score = round(sum(sorted(scores)[:n_worst]) / n_worst, 2)

        performance = {
            "combined_score": worst_avg_score,
            "windows": window_performances,
        }
        errors = [p["error"] for p in window_performances if p.get("error")]
        if errors:
            performance["error"] = "\n\n".join(errors)

        strategy.performance = performance
        with open(strategy.strategy_dir / "metrics.json", "w") as f:
            json.dump(performance, f, indent=4)

        logger.info(f"Strategy {strategy.id} window performance: {performance}")
        return performance
