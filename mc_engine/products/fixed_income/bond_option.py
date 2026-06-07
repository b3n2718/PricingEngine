import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData


class BondOption(Product):
    """European call option on a zero-coupon bond.

    At option expiry T_option the holder may buy a ZCB maturing at T_bond for
    the strike price K.  The payoff is:

        Payoff = D(0, T_opt) · max(N · P(T_opt, T_bond) - K, 0)

    where D(0, T_opt) is the stochastic discount factor and P(T_opt, T_bond)
    is computed analytically from the short-rate path via the affine formula.

    Parameters
    ----------
    underlying:
        Asset identifier for the interest-rate process.
    notional:
        Face value N of the underlying zero-coupon bond.
    maturity_option:
        Expiry of the option in years (T_opt).
    maturity_bond:
        Maturity of the underlying bond in years (T_bond > T_opt).
    strike:
        Option strike K (price of the bond at option expiry).
    """

    def __init__(self, underlying: str, notional: float,
                 maturity_option: float, maturity_bond: float,
                 strike: float):
        self._underlying    = underlying
        self._maturity      = maturity_option
        self._maturity_bond = maturity_bond
        self._notional      = notional
        self._strike        = strike

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        """Discounted option payoff across all paths.

        Parameters
        ----------
        paths:
            Path data implementing both ``df()`` and ``zcp(t1, t2)``.
        discount:
            Unused — discounting is done via ``df()``.

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        p = paths[self._underlying]
        bond_price = p.zcp(self._maturity, self._maturity_bond) * self._notional
        return p.df() * np.maximum(bond_price - self._strike, 0)
