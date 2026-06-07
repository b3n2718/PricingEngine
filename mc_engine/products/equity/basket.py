import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData
from scipy.stats import gmean


class BasketOption(Product):
    """Basket option on a weighted portfolio of equities.

    The payoff is based on the weighted average terminal price across all
    underlying assets:

        Arithmetic:  S_basket = Σ wᵢ · Sᵢ(T)
        Geometric:   S_basket = Π Sᵢ(T)^wᵢ  (weighted geometric mean)

        Payoff = D(0, T) · max(φ · (S_basket - K), 0)

    Parameters
    ----------
    underlying:
        List of asset identifiers — must match keys in the ``processes`` dict.
    strike:
        Basket option strike K.
    maturity:
        Time to expiry in years.
    is_call:
        True for a call, False for a put.
    arithmetic:
        True (default) for weighted arithmetic average, False for geometric.
    weights:
        Optional dict mapping asset id to weight.  If None, equal weights are
        used.  Custom weights are normalised to sum to 1.
    """

    def __init__(self, underlying: list, strike: float,
                 maturity: float, is_call: bool,
                 arithmetic: bool = True, weights: dict = None):
        self._underlying = underlying
        self.strike      = strike
        self._maturity   = maturity
        self.is_call     = is_call
        self.mean        = "arithmetic" if arithmetic else "geometric"

        if weights is None:
            # Equal-weight basket
            self.weights = {key: 1 / len(underlying) for key in underlying}
        else:
            total = sum(weights.values())
            self.weights = {key: w / total for key, w in weights.items()}

    @property
    def underlyings(self) -> list[str]:
        return self._underlying

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        """Compute the discounted basket payoff across all paths.

        Parameters
        ----------
        paths:
            Terminal path values for each underlying asset.
        discount:
            Deterministic discount factor D(0, T).

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        S_T     = [paths[key].terminal_value() for key in self._underlying]
        weights = [self.weights[key] for key in self._underlying]

        if self.mean == "arithmetic":
            basket = np.sum(
                np.array(weights)[:, None] * np.array(S_T), axis=0
            )
        else:
            basket = gmean(
                np.array(S_T), axis=0,
                weights=np.array(weights)[:, None]
            )

        sign = 1.0 if self.is_call else -1.0
        return discount * np.maximum(sign * (basket - self.strike), 0.0)
