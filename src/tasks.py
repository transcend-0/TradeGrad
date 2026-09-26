"""Task registry: one entry per (market, mode), shared by training and evaluation.

A task fixes the market (backtest settings, trade-frequency floor, history fields), the seed
strategy TradeTGD starts from, and the two backtest windows: 2018-2023 (six yearly windows,
worst-30% CVaR) for training, 2024-2025 for the held-out test score. Everything else -- the
optimizer's hyperparameters, the step budget -- is a run-time argument, not part of the task.
"""
from pathlib import Path

from .market import MARKETS

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent

TRAIN_START, TRAIN_END = "2018-01-01", "2023-12-31"
TEST_START, TEST_END = "2024-01-01", "2025-12-31"

TASKS = {}
for _mkt in ("CN", "US"):
    _m = MARKETS[_mkt]
    TASKS[f"{_mkt.lower()}_time"] = dict(
        market=_mkt,
        mode="time",
        result_root=str(REPO_ROOT / "result" / _mkt / "time"),
        log_root=str(REPO_ROOT / "log" / _mkt / "time"),
        evaluate=str(SRC_DIR / "evaluate_time.py"),
        initial=_m["initial_time"],
        loss_kwargs=dict(data_path=_m["index_path"]),
    )
    TASKS[f"{_mkt.lower()}_cross"] = dict(
        market=_mkt,
        mode="cross",
        result_root=str(REPO_ROOT / "result" / _mkt / "cross"),
        log_root=str(REPO_ROOT / "log" / _mkt / "cross"),
        evaluate=str(SRC_DIR / "evaluate.py"),
        initial=_m["initial_cross"],
        loss_kwargs=dict(date2symbol_dir=_m["date2symbol_dir"], data_dir=_m["data_dir"]),
    )
