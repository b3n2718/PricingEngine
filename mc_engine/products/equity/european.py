import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData


class EuropeanOption(Product):
    """Standard European vanilla call or put option.

    The payoff is exercised only at maturity T:

        Payoff = D(0, T) · max(φ · (S(T) - K), 0)

    where φ = +1 for a call and φ = -1 for a put.

    Parameters
    ----------
    underlying:
        Asset identifier matching the key in the ``processes`` dict.
    strike:
        Option strike price K.
    maturity:
        Time to expiry in years.
    is_call:
        True for a call option, False for a put.
    """

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
        """Compute the discounted European payoff across all paths.

        Parameters
        ----------
        paths:
            Path data for the underlying asset.
        discount:
            Deterministic discount factor D(0, T).

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        S_T  = paths[self._underlying].terminal_value()
        sign = 1.0 if self.is_call else -1.0
        return discount * np.maximum(sign * (S_T - self.strike), 0.0)
