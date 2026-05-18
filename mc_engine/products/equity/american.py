import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData

class AmericanOption(Product):

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
        S  = paths[self._underlying].full_price_path()
        V = np.maximum(S - self.strike, 0)
        dt = self._maturity/S.shape[1]
        
        discount_rate = -np.log(discount)/self._maturity
        dcf = np.exp(-discount_rate * dt)

        for i in range(S.shape[1]-1):

            t = -(i + 2)
            t_next = -(i + 1)

            S_t = S[:, t]
            V_next = V[:, t_next]

            itm = V_next > 0   # oder S_t < K

            if np.sum(itm) == 0:
                continue

            Y = V_next[itm] * dcf

            coeff = np.polyfit(S_t[itm], Y, 2)

            C = np.polyval(coeff, S_t)

            E = np.maximum(S_t - self.strike, 0)

            exercise = E > C

            V[:, t] = np.where(exercise, E, V_next * dcf)
        return V[:,0]

