import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData


class Swaption(Product):
    """Caplet — a call on the LIBOR/forward rate over a single period.

    A caplet pays:

        Payoff = N · δ · D(0, T_end) · max(L(T_start, T_end) - K, 0)

    The forward rate L is obtained from the bond price relation:

        1 + δ · L(T_start, T_end) = 1 / P(T_start, T_end)

    so the payoff can be re-expressed using bond prices, avoiding direct
    rate simulation.

    Parameters
    ----------
    underlying:
        Asset identifier for the interest-rate process.
    notional:
        Notional N of the caplet.
    maturity:
        Start of the caplet period T_start in years.
    period:
        Length of the caplet period δ in years (T_end = maturity + period).
    strike:
        Cap strike rate K.
    """

    def __init__(self, underlying: str, notional: float,
                 maturity: float, period: float, strike: float):
        self._underlying = underlying
        self._maturity   = maturity
        self._period     = period
        self._strike     = strike
        self._notional   = notional

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        """Compute the discounted caplet payoff across all paths.

        Uses the bond-price representation so that only ``df()`` and
        ``zcp(t1, t2)`` are required from the path container.

        Parameters
        ----------
        paths:
            Path data implementing ``df()`` and ``zcp(t1, t2)``.
        discount:
            Unused — discounting is done via ``df()``.

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        p = paths[self._underlying]
        T_end  = self._maturity + self._period
        P      = p.zcp(self._maturity, T_end)

        # max(1/P - (1 + δ·K), 0) is equivalent to δ·max(L - K, 0) / (1+δK)
        return p.df() * np.maximum(1.0 / P - (1.0 + self._period * self._strike), 0.0)
