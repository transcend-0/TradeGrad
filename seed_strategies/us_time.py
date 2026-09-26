import numpy as np
from typing import Any
from akquant import Strategy


class TimeStrategy(Strategy):

    def __init__(self):
        super().__init__()
        self.target_symbol = "target_symbol"

    def on_start(self):
        self.subscribe(self.target_symbol)
        self.log("on_start")

    def on_bar(self, bar):
### EDIT START
        fast_window = 5
        slow_window = 20

        close = self.get_history(count=slow_window, symbol=self.target_symbol, field="close")

        fast_ma = close[-fast_window:].mean()
        slow_ma = close.mean()

        target_percent = 0.95 if fast_ma > slow_ma else -0.95
        self.order_target_percent(symbol=self.target_symbol, target_percent=target_percent)
### EDIT END
