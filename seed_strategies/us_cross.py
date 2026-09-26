import numpy as np
from typing import Any
from akquant import Strategy


class AlphaStrategy(Strategy):

    def __init__(self):
        super().__init__()
        self.top_n = 30
        self.bottom_n = 30

    def on_start(self):
        for symbol in self.all_symbols:
            self.subscribe(symbol)
        self.log(f"on_start")

    def on_cross_section(self, trading_date, timestamp):
        symbols = self.date2symbols[trading_date]

        scores = self.calculate_scores(symbols)

        long_selected, short_selected = self.rebalance_long_short(scores)
        self.log(f"rebalance long={long_selected} short={short_selected}")

    def rebalance_long_short(
        self, scores: dict[str, float]
    ) -> tuple[list[str], list[str]]:
        ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        long_selected = [symbol for symbol, _ in ranking[: self.top_n]]
        short_selected = [
            symbol
            for symbol, _ in ranking[-self.bottom_n :]
            if symbol not in long_selected
        ] if self.bottom_n > 0 else []

        prices = self.get_history_map(
            count=1,
            symbols=long_selected + short_selected,
            field="close",
        )
        equity = self.equity

        target_positions: dict[str, float] = {}
        if long_selected:
            long_weight = 0.45 / len(long_selected)
            for symbol in long_selected:
                target_positions[symbol] = long_weight * equity / prices[symbol][0]
        if short_selected:
            short_weight = -0.45 / len(short_selected)
            for symbol in short_selected:
                target_positions[symbol] = short_weight * equity / prices[symbol][0]

        self.rebalance_positions(
            target_positions=target_positions,
            liquidate_unmentioned=True,
            allow_short=True,
        )
        return long_selected, short_selected

    def calculate_scores(self, symbols: list[str]) -> dict[str, float]:
### EDIT START
        history_map_close = self.get_history_map(
            count=3,
            symbols=symbols,
            field="close",
        )
        # Pad to 2D array for vectorized operations: [num_symbols, num_days]
        close = np.array([history_map_close[symbol] for symbol in symbols])
        alpha = close[:, 0] / close[:, -1] - 1
        scores = dict(zip(symbols, alpha))
        return scores
### EDIT END
