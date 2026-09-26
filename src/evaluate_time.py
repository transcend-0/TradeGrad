import os
import argparse
import json
import math
import pandas as pd
import akquant as aq

try:
    from .market import MARKETS, backtest_kwargs
except ImportError:  # run as a script by the loss layer
    from market import MARKETS, backtest_kwargs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy_path", type=str, help="Path to the trading strategy source code.")
    parser.add_argument("--result_dir", type=str, help="Directory to save strategy performance results.")
    parser.add_argument("--data_path", type=str, help="Path to market data csv (single symbol).")
    parser.add_argument("--symbol", type=str, default="target_symbol", help="Symbol name used inside the strategy/data.")
    parser.add_argument("--start_time", type=str, help="Start time for backtesting.")
    parser.add_argument("--end_time", type=str, help="End time for backtesting.")
    return parser.parse_args()

def load_data(data_path: str, symbol: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(data_path, parse_dates=["date"])
    df["symbol"] = symbol
    return {symbol: df}

# Trade-frequency floor for the cross-sectional task: 0.25x the initial strategy's own trade rate
# (see market.py's freq_floor). Loose enough to leave room for genuinely lower-turnover designs,
# tight enough to kill the reward-hacking route of trading less to flatten drawdown.
MARKET = os.environ.get("AKQUANT_MARKET", "CN")
_floor = MARKETS[MARKET]["freq_floor"]["time"]
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
    data_path: str,
    symbol: str,
    start_time: str,
    end_time: str,
):
    data_map = load_data(data_path, symbol)

    result = aq.run_backtest(
        data=data_map,
        strategy_source=strategy_path,
        context={
            "target_symbol": symbol,
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
