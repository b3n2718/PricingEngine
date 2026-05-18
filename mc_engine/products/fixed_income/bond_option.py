import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData

class BondOption(Product):

    def __init__(self, underlying: str,notional: float,
                 maturity_option: float, maturity_bond: float,strike:float):
        self._underlying = underlying
        self._maturity   = maturity_option
        self._maturity_bond = maturity_bond
        self._notional   = notional
        self._strik = strike

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        return  paths[self._underlying].df() * np.maximum(paths[self._underlying].zcp(self._maturity,self._maturity_bond) * self._notional - self._strik,0)

