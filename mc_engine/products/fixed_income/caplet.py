import numpy as np
from products.base import Product
from paths.base import PathData

class Swaption(Product):

    def __init__(self, underlying: str,notional: float,
                 maturity: float,period: float, strike:float):
        self._underlying = underlying
        self._maturity   = maturity
        self._period     = period
        self._strike     = strike

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
            discount: float) -> np.ndarray:

        P = paths[self._underlying].zcp(self._maturity, self._maturity+self._period)

        return paths[self._underlying].df() * np.maximum(1/P-(1+self._period*self._strike), 0.0)
