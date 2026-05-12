import numpy as np
from products.base import Product
from paths.base import PathData

class ZeroCouponBond(Product):

    def __init__(self, underlying: str,notional: float,
                 maturity: float):
        self._underlying = underlying
        self._maturity   = maturity
        self._notional   = notional

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        return paths[self._underlying].df() * self._notional

