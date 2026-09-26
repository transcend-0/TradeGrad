import os
import re
import pickle
import argparse
import time
import glob
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import bisect
import pandas as pd
import datetime
import akquant as aq

try:
    from .market import MARKETS, backtest_kwargs
except ImportError:  # run as a script by the loss layer
    from market import MARKETS, backtest_kwargs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy_path", type=str, help="Path to the trading strategy source code.")
    parser.add_argument("--result_dir", type=str, help="Directory to save strategy performance results.")
    parser.add_argument("--date2symbol_dir", type=str, help="Path to date2symbol mapping file.")
    parser.add_argument("--data_dir", type=str, help="Path to market data directory.")
    parser.add_argument("--start_time", type=str, help="Start time for backtesting.")
    parser.add_argument("--end_time", type=str, help="End time for backtesting.")
    return parser.parse_args()

def _read_one_csv(path: str):
    symbol = Path(path).stem
    df = pd.read_csv(path, parse_dates=["date"])
    df["symbol"] = symbol
    return symbol, df

def _cache_path(data_dir: str, symbols: list[str]) -> Path:
    """Cache file keyed by the symbol set and the newest source file, so it self-invalidates."""
    import hashlib
    d = Path(data_dir)
    newest = max((p.stat().st_mtime for p in d.glob("*.csv")), default=0)
    key = hashlib.md5(f"{d.resolve()}|{newest}|{','.join(sorted(symbols))}".encode()).hexdigest()
    cache_dir = Path(os.environ.get("AKQUANT_DATA_CACHE", "/tmp/akquant_data_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.pkl"


def load_data(
    data_dir: str,
    max_workers: int | None = None,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read the per-symbol csvs.

    `symbols` restricts the read to the symbols actually used. The caller filters to the index
    constituents immediately afterwards anyway, and the data directory holds far more symbols than
    any one universe uses, so reading all of them throws away most of the work on every single
    evaluation -- which matters because each evaluation is its own process.

    Results are also pickled, keyed by the symbol set and the newest csv mtime, so repeated
    evaluations over the same universe skip csv parsing entirely.
    """
    time_start = time.time()

    if symbols is not None:
        cache = _cache_path(data_dir, symbols)
        if cache.exists():
            try:
                with open(cache, "rb") as f:
                    data_map = pickle.load(f)
                print(f"Loaded {len(data_map)} symbols from cache in "
                      f"{time.time() - time_start:.2f} seconds")
                return data_map
            except Exception:
                pass  # fall through and rebuild
        paths = [str(Path(data_dir) / f"{symbol}.csv") for symbol in symbols]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        cache = None
        paths = sorted(glob.glob(str(Path(data_dir) / "*.csv")))

    if max_workers is None:
        max_workers = os.cpu_count() or 1
        max_workers = min(max_workers, len(paths))

    data_map: dict[str, pd.DataFrame] = {}
    if max_workers <= 1 or len(paths) == 1:
        for path in paths:
            symbol, df = _read_one_csv(path)
            data_map[symbol] = df
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_read_one_csv, path) for path in paths]
            for future in as_completed(futures):
                try:
                    symbol, df = future.result()
                    data_map[symbol] = df
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue

    if symbols is not None and cache is not None:
        try:
            tmp = cache.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(data_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache)   # atomic, so concurrent evaluations never read a partial file
        except Exception:
            pass

    print(f"Loaded {len(data_map)} symbols in {time.time() - time_start:.2f} seconds (workers={max_workers})")
    return data_map


class Date2Symbols(dict[int, set[str]]):
    """
    dict: {update_date -> set(symbols)}
    - key uses int form yyyymmdd (e.g. 20151130)
    - query: d[any_date] returns the constituent set as of the most recent rebalance <= any_date
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._sorted_keys: list[int] | None = None
        if args or kwargs:
            self.update(*args, **kwargs)

    @staticmethod
    def _norm_date(value) -> int:
        """Normalize input into int yyyymmdd.

        Query supports: str, datetime.date.
        Also tolerates: datetime.datetime, pandas.Timestamp, int.
        """
        if isinstance(value, int):
            # Prefer treating as yyyymmdd if it looks like a date.
            if 19000101 <= value <= 21000101:
                return value

            # Otherwise treat as epoch timestamp (ns/ms/s heuristic)
            abs_v = abs(value)
            if abs_v >= 10**14:
                ts = pd.to_datetime(value, unit="ns", utc=True)
            elif abs_v >= 10**11:
                ts = pd.to_datetime(value, unit="ms", utc=True)
            else:
                ts = pd.to_datetime(value, unit="s", utc=True)
            if getattr(ts, "tz", None) is not None:
                ts = ts.tz_convert(None)
            d = ts.normalize().date()
            return d.year * 10000 + d.month * 100 + d.day

        if isinstance(value, datetime.datetime):
            d = value.date()
            return d.year * 10000 + d.month * 100 + d.day
        if isinstance(value, datetime.date):
            return value.year * 10000 + value.month * 100 + value.day

        ts = pd.Timestamp(value)
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert(None)
        d = ts.normalize().date()
        return d.year * 10000 + d.month * 100 + d.day

    def _ensure_sorted_keys(self) -> list[int]:
        if self._sorted_keys is None:
            self._sorted_keys = sorted(self.keys())
        return self._sorted_keys

    def effective_date(self, query_date) -> int:
        q = self._norm_date(query_date)
        keys = self._ensure_sorted_keys()
        if not keys:
            raise KeyError("Date2Symbols is empty")

        i = bisect.bisect_right(keys, q) - 1

        return None if i < 0 else keys[i]

    def as_of(self, query_date) -> set[str]:
        k = self.effective_date(query_date)
        return dict.__getitem__(self, k)

    def __missing__(self, key):
        return self.as_of(key)

    def __setitem__(self, key, value) -> None:
        k = self._norm_date(key)
        dict.__setitem__(self, k, set(value))
        self._sorted_keys = None

    def update(self, *args, **kwargs) -> None:
        other = dict(*args, **kwargs)
        for k, v in other.items():
            self[k] = v
        self._sorted_keys = None

    @classmethod
    def from_date2symbol_dir(cls, date2symbol_dir: str | Path) -> "Date2Symbols":
        """
        Reads data/<index>_list/*.csv into {updateDate -> set(code)}. Expected columns:
        updateDate, code, code_name.
        """
        date2symbol_dir = Path(date2symbol_dir)
        paths = sorted(date2symbol_dir.glob("*_list_*.csv"))
        merged = {}

        for path in paths:
            df = pd.read_csv(path)

            # Which column carries the snapshot date differs by market. The HS300 files stamp every
            # row with that file's rebalance date, so `updateDate` is the snapshot. The S&P files
            # stamp each row with the date that company joined the index -- grouping on it would
            # produce hundreds of "snapshots" holding one or two tickers each. There the snapshot
            # is the month in the file name, and the whole file is the membership on that date.
            stamped = pd.to_datetime(df["updateDate"], errors="coerce").dt.normalize()
            codes = df["code"].astype(str).sort_values()
            if stamped.nunique(dropna=True) > 1:
                m = re.search(r"_(\d{4})-(\d{2})", path.name)
                if m is None:
                    raise ValueError(f"cannot read a snapshot date from {path.name}")
                snapshot = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1).normalize()
                stamped = pd.Series(snapshot, index=df.index)

            for d, group_codes in codes.groupby(stamped):
                if pd.isna(d):
                    continue
                dd_ts = pd.Timestamp(d)
                if getattr(dd_ts, "tz", None) is not None:
                    dd_ts = dd_ts.tz_convert(None)
                dd = dd_ts.normalize().date()
                key = dd.year * 10000 + dd.month * 100 + dd.day
                group_codes = sorted(set(group_codes.tolist()))
                merged.setdefault(key, set()).update(group_codes)

        for k, v in merged.items():
            merged[k] = list(v)

        return cls(merged)


# Trade-frequency floor for the cross-sectional task: 0.25x the initial strategy's own trade rate
# (see market.py's freq_floor). Loose enough to leave room for genuinely lower-turnover designs,
# tight enough to kill the reward-hacking route of trading less to flatten drawdown.
MARKET = os.environ.get("AKQUANT_MARKET", "CN")
_floor = MARKETS[MARKET]["freq_floor"]["cross"]
FREQ_FLOOR = float(os.environ.get("AKQUANT_FREQ_FLOOR", _floor if _floor is not None else 0.0))
FREQ_SHARPNESS = float(os.environ.get("AKQUANT_FREQ_SHARPNESS", 1.0))
# Weight on the AR term vs MDD in reward_function's geometric mean; 0.5 = equal weight, raising it trades less emphasis on drawdown for more on return.
AR_WEIGHT = float(os.environ.get("AKQUANT_AR_WEIGHT", 0.5))


def reward_function(
    metrics: dict,
    ar_weight: float = AR_WEIGHT,
    ar_reference: float = 20,
    mdd_reference: float = 15,
) -> float:
    ar_score = 1.0 / (
        1.0 + math.exp(-0.11 * (metrics['annualized_return'] - ar_reference))
    )
    mdd_score = 1.0 / (
        1.0 + math.exp(0.22 * (metrics['max_drawdown'] - mdd_reference))
    )
    base = ar_score ** ar_weight * mdd_score ** (1 - ar_weight)

    gate = 1.0
    if FREQ_FLOOR > 0:
        gate = min(1.0, metrics['trade_per_day'] / FREQ_FLOOR) ** FREQ_SHARPNESS

    return 100 * base * gate


def run_backtest(
    strategy_path: str,
    result_dir: str,
    date2symbol_dir: str,
    data_dir: str,
    start_time: str,
    end_time: str,
):
    date2symbols = Date2Symbols.from_date2symbol_dir(date2symbol_dir)
    all_symbols = set()
    for symbols in date2symbols.values():
        all_symbols.update(symbols)
    all_symbols = list(sorted(all_symbols))

    data_map = load_data(data_dir, max_workers=12, symbols=all_symbols)
    data_map = {symbol: data_map[symbol] for symbol in all_symbols if symbol in data_map}

    result = aq.run_backtest(
        data=data_map,
        strategy_source=strategy_path,
        context={
            "all_symbols": all_symbols,
            "date2symbols": date2symbols,
        },
        start_time=start_time,
        end_time=end_time,
        **backtest_kwargs(MARKET),
    )

    metrics_df = result.metrics_df

    metrics = {
        'sharpe_ratio':  metrics_df.loc['sharpe_ratio', 'value'],
        'annualized_return': metrics_df.loc['annualized_return', 'value'] * 100,
        'max_drawdown': metrics_df.loc['max_drawdown_pct', 'value'],
        'trade_per_day': metrics_df.loc['closed_trade_count', 'value'] / (metrics_df.loc["duration", "value"].days / 365 * 243),
        'win_rate': metrics_df.loc['win_rate', 'value'],
        'return_per_trade': metrics_df.loc['avg_return_pct', 'value'],
    }

    if metrics['trade_per_day'] == 0:
        raise ValueError("No trades executed, invalid strategy.")

    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            metrics[k] = round(v, 2)

    performance = {
        "combined_score": round(reward_function(metrics), 2),
        "metrics": metrics
    }

    with open(os.path.join(result_dir, "metrics.json"), 'w') as f:
        json.dump(performance, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    run_backtest(**vars(args))
