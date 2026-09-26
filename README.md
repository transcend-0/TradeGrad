# TradeGrad

This is the implementation of our paper *Trading Strategy Optimization via Textual Gradient*.

TradeGrad uses an LLM to iteratively rewrite a trading strategy's source code, guided by textual
feedback derived from its own backtest performance. The optimizer, **TradeTGD**, samples a
reference set from the score-ranked strategies tried so far, writes a textual gradient against
that set plus a running memory summary, picks the next base program uniformly from the
references, and
revises it as either a small local edit or a full rewrite. Training is scored by the worst-30%
CVaR over yearly backtest windows, so the search can't win by blowing up in a bad
year; a trade-frequency gate rules out the degenerate "stop trading to flatten drawdown" solution.

## Layout

```
src/               optimizer (TradeTGD), loss/reward, backtest evaluators, task registry, run loop
seed_strategies/   initial strategy each task starts from
script/            train.sh / eval.sh -- entry points, see below
```

## Setup

```
pip install -r requirements.txt
```

Two things live outside this repo and need to be pointed at on your machine:

- **`OPENROUTER_API_KEY`** -- or any OpenAI-compatible key (see `--base-url`/`--model` on `run_one.py`)
- **`AKQUANT_DATA_ROOT`** -- per-symbol OHLCV csvs and index constituent lists (default: a `data/`
  directory next to this repo). Expected layout:
  ```
  <AKQUANT_DATA_ROOT>/CN/<symbol>.csv         <AKQUANT_DATA_ROOT>/US/<symbol>.csv
  <AKQUANT_DATA_ROOT>/CN_index/sh.000300.csv  <AKQUANT_DATA_ROOT>/US_index/SP500.csv
  <AKQUANT_DATA_ROOT>/hs300_list/*_list_*.csv <AKQUANT_DATA_ROOT>/SP500_list/*_list_*.csv
  ```

## Running

`script/train.sh` and `script/eval.sh` hold their settings as plain variables at the top of the
file (`YOUR_DATA_ROOT` / `YOUR_API_KEY` placeholders included) -- edit them, then:

```
bash script/train.sh      # train a task, then score it on the held-out 2024-2025 window
bash script/eval.sh       # re-score an existing run without retraining
```

Both refuse to run while a placeholder is unedited, and both launch detached (`nohup`, backgrounded)
so the run survives the terminal disconnecting. `script/my_train.sh` / `script/my_eval.sh` are
gitignored personal copies with your machine's settings filled in, for everyday use.

Each run writes to `result/<MARKET>/<mode>/checkpoint/<step>/`, `checkpoint_test/<step>/` (the
same checkpoints scored on the test window), and `best/` (the highest-scoring checkpoint once
training finishes). Neither `result/` nor `log/` is tracked by git.
