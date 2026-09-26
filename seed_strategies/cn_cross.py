import numpy as np
from typing import Any
from akquant import Strategy


class AlphaStrategy(Strategy):

    def __init__(self):
        super().__init__()
        self.top_n = 30

    def on_start(self):
        for symbol in self.all_symbols:
            self.subscribe(symbol)
        self.log(f"on_start")

    def on_cross_section(self, trading_date, timestamp):
        symbols = self.date2symbols[trading_date]

        scores = self.calculate_scores(symbols)

        selected = self.rebalance_to_topn(
            scores=scores,
            top_n=self.top_n,
            weight_mode="equal",
            long_only=False,
            liquidate_unmentioned=True,
        )
        self.log(f"rebalance selected={selected}")

    def calculate_scores(self, symbols: list[str]) -> dict[str, float]:
### EDIT START
        history_map_close = self.get_history_map(
            count=3,
            symbols=symbols,
            field="close",
        )
        # Pad to 2D array for vectorized operations: [num_symbols, num_days]
        close = np.array([history_map_close[symbol] for symbol in symbols])
        alpha = (close[:, -1] - close[:, 0]) / close[:, 0]
        scores = dict(zip(symbols, alpha))
        return scores
### EDIT END