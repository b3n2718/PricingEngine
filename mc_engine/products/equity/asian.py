import numpy as np
from products.base import Product
from paths.base import PathData
from scipy.stats import gmean

class AsianOption(Product):

    def __init__(self, underlying: str, strike: float,
                 maturity: float, is_call: bool, arithmetic:bool=True):
        self._underlying = underlying
        self.strike      = strike
        self._maturity   = maturity
        self.is_call     = is_call
        self.mean  = "arithmetic" if arithmetic else "geometric"

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        S_T  = paths[self._underlying].full_price_path()
        sign = 1.0 if self.is_call else -1.0
        if self.mean == "arithmetic":
            S_T = np.mean(S_T,axis=1)
        elif "geometric":
            S_T = gmean(S_T,axis=1)
                
        
        return discount * np.maximum(sign * (S_T - self.strike), 0.0)

