import numpy as np
from products.base import Product
from paths.base import PathData
from scipy.stats import gmean

class BasketOption(Product):

    def __init__(self, underlying: list, strike: float,
                 maturity: float, is_call: bool, arithmetic:bool=True, weights:dict=None):
        self._underlying = underlying
        self.strike      = strike
        self._maturity   = maturity
        self.is_call     = is_call
        self.mean  = "arithmetic" if arithmetic else "geometric"
        if weights is None:
            self.weights = {key: 1/len(underlying) for key in underlying}
        else: 
            self.weights = {key: weight/sum(weights.values) for key,weight in weights.values()}
            

    @property
    def underlyings(self) -> list[str]:
        return self._underlying

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        S_T = list()
        weights = list()
        for key in self._underlying:
            S_T.append(paths[key].terminal_value())
            weights.append(self.weights[key])

        if self.mean == "arithmetic":
            S_T = np.sum(np.array(weights)[:,None] * np.array(S_T),axis=0)
        elif "geometric":
            S_T = gmean(np.array(S_T), axis = 0,weights=np.array(weights)[:,None])
                
        sign = 1.0 if self.is_call else -1.0
        return discount * np.maximum(sign * (S_T - self.strike), 0.0)

