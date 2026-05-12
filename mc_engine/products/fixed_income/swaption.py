import numpy as np
from products.base import Product
from paths.base import PathData

class Swaption(Product):

    def __init__(self, underlying: str,notional: float,
                 maturity_option: float, fixed_rate: float,freq:float,swap_maturity:float):
        self._underlying = underlying
        self._maturity   = maturity_option
        self._fixed_rate = fixed_rate
        self._freq       = freq
        self._swap_maturity    = swap_maturity
        self._notional   = notional

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
            discount: float) -> np.ndarray:

        T0 = self._maturity
        Tn = self._maturity + self._swap_maturity
        delta_i = 1 / self._freq

        # Floating leg
        floating_leg = 1.0 - paths[self._underlying].zcp(T0, Tn)

        # Fixed leg
        fixed_leg = 0.0
        n = int(self._swap_maturity * self._freq)

        for i in range(1, n + 1):
            Ti = T0 + i * delta_i
            fixed_leg += self._fixed_rate * delta_i * paths[self._underlying].zcp(T0, Ti)

        V_t = floating_leg - fixed_leg

        return paths[self._underlying].df() * self._notional * np.maximum(V_t, 0.0)
