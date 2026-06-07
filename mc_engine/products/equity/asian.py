import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData
from scipy.stats import gmean


class AsianOption(Product):
    """Asian (average-price) call or put option.

    The payoff depends on the average spot price over the life of the option
    rather than the terminal value alone:

        Arithmetic:  Payoff = D(0,T) · max(φ · (mean(S) - K), 0)
        Geometric:   Payoff = D(0,T) · max(φ · (geomean(S) - K), 0)

    Parameters
    ----------
    underlying:
        Asset identifier matching the key in the ``processes`` dict.
    strike:
        Option strike price K.
    maturity:
        Time to expiry in years.
    is_call:
        True for a call, False for a put.
    arithmetic:
        True (default) for arithmetic average, False for geometric average.
    """

    def __init__(self, underlying: str, strike: float,
                 maturity: float, is_call: bool, arithmetic: bool = True):
        self._underlying = underlying
        self.strike      = strike
        self._maturity   = maturity
        self.is_call     = is_call
        self.mean        = "arithmetic" if arithmetic else "geometric"

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        """Compute the discounted Asian payoff across all paths.

        Parameters
        ----------
        paths:
            Full price path for the underlying asset.
        discount:
            Deterministic discount factor D(0, T).

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        S_T  = paths[self._underlying].full_price_path()   # [n_sims, n_steps]
        sign = 1.0 if self.is_call else -1.0

        if self.mean == "arithmetic":
            S_avg = np.mean(S_T, axis=1)
        else:
            S_avg = gmean(S_T, axis=1)

        return discount * np.maximum(sign * (S_avg - self.strike), 0.0)
