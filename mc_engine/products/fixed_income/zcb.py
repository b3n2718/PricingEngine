import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData


class ZeroCouponBond(Product):
    """Zero-coupon bond priced by Monte Carlo.

    The bond pays a fixed notional at maturity T.  Its present value is the
    expected stochastic discount factor, computed from the simulated short-rate
    (or forward-rate) paths:

        Price = N · E[exp(-∫₀ᵀ r(t) dt)]

    Parameters
    ----------
    underlying:
        Asset identifier for the interest-rate process driving the discount.
    notional:
        Face value N paid at maturity.
    maturity:
        Bond maturity in years.
    """

    def __init__(self, underlying: str, notional: float, maturity: float):
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
        """Discounted notional using the stochastic discount factor.

        Parameters
        ----------
        paths:
            Path data for the interest-rate process (must implement ``df()``).
        discount:
            Unused — ZCB uses the path's own stochastic discount factor.

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        return paths[self._underlying].df() * self._notional
