"""Market-specific backtest settings and the trade-frequency floor, in one place.

`freq_floor` is a quarter of the initial strategy's own trade rate on that market and task. The
reward multiplies by `min(1, trade_per_day / freq_floor)`, which is exactly 1 above the floor, so
a strategy that trades normally is scored as if the gate were not there; only the degenerate
"stop trading to flatten drawdown" route is priced.

Market data (per-symbol OHLCV csvs, index constituent lists) is not part of this repo. It is
expected under `AKQUANT_DATA_ROOT` (default: a `data/` directory next to this repo) -- see the
top-level README for the expected layout. Point that env var elsewhere to run against a different
copy of the data.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = REPO_ROOT / "seed_strategies"
DATA_ROOT = Path(os.environ.get("AKQUANT_DATA_ROOT", REPO_ROOT / "data"))

MARKETS = {
    "CN": dict(
        commission_rate=0.0005,
        stamp_tax_rate=0.001,      # sell-side stamp duty, A-share only
        slippage={"type": "percent", "value": 0.0005},
        t_plus_one=True,           # A-shares cannot be sold the day they are bought
        index_path=str(DATA_ROOT / "CN_index/sh.000300.csv"),
        data_dir=str(DATA_ROOT / "CN"),
        date2symbol_dir=str(DATA_ROOT / "hs300_list"),
        # A-share csvs carry turnover value and turnover rate on top of OHLCV.
        fields=("open", "high", "low", "close", "volume", "amount", "turn"),
        initial_time=str(SEED_DIR / "cn_time.py"),
        initial_cross=str(SEED_DIR / "cn_cross.py"),
        # 0.25x the seed strategy's mean trade rate over 2018-2023.
        freq_floor=dict(time=0.0125, cross=4.1075),
    ),
    "US": dict(
        commission_rate=0.0005,
        stamp_tax_rate=0.0,        # no stamp duty
        slippage={"type": "percent", "value": 0.0005},
        t_plus_one=False,          # same-day round trips are allowed
        # The seed US cross-sectional strategy is long-short (`rebalance_long_short`), and the
        # engine rejects negative targets unless the risk config advertises short-sell support.
        risk=dict(account_mode="margin", enable_short_sell=True),
        index_path=str(DATA_ROOT / "US_index/SP500.csv"),
        data_dir=str(DATA_ROOT / "US"),
        date2symbol_dir=str(DATA_ROOT / "SP500_list"),
        # The US csvs are OHLCV only -- naming `amount` or `turn` here would send the model
        # after columns that do not exist.
        fields=("open", "high", "low", "close", "volume"),
        initial_time=str(SEED_DIR / "us_time.py"),
        initial_cross=str(SEED_DIR / "us_cross.py"),
        freq_floor=dict(time=0.0187, cross=13.4854),
    ),
}


def backtest_kwargs(market: str) -> dict:
    m = MARKETS[market]
    extra = {}
    if m.get("risk"):
        # Passed as a plain dict on purpose. The engine's RiskConfig branch reads
        # `risk_config.__dataclass_fields__`, which the Rust binding does not expose, so handing it
        # a RiskConfig instance fails silently and shorting stays disabled; the dict branch sets the
        # fields one by one and works.
        extra["risk_config"] = dict(m["risk"])
    return dict(
        **extra,
        initial_cash=1_000_000.0,
        commission_rate=m["commission_rate"],
        stamp_tax_rate=m["stamp_tax_rate"],
        slippage=m["slippage"],
        history_depth=252,
        t_plus_one=m["t_plus_one"],
    )
