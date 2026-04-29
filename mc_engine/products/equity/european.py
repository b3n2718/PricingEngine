import numpy as np
from products.base import Product
from paths.base import PathData

class EuropeanOption(Product):

    def __init__(self, underlying: str, strike: float,
                 maturity: float, is_call: bool):
        self._underlying = underlying
        self.strike      = strike
        self._maturity   = maturity
        self.is_call     = is_call

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        S_T  = paths[self._underlying].terminal_value()
        sign = 1.0 if self.is_call else -1.0
        return discount * np.maximum(sign * (S_T - self.strike), 0.0)

