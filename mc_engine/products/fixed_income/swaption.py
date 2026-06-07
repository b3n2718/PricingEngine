import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData


class Swaption(Product):
    """Payer swaption on an interest-rate swap.

    The holder has the right to enter a payer (fixed-for-floating) swap at
    option expiry T₀.  The swap runs from T₀ to T₀ + swap_maturity with
    payment frequency ``freq`` per year.

    Using the annuity as numeraire, the payoff at T₀ can be written as:

        V(T₀) = N · max(floating_leg - fixed_leg, 0)

    where:
        floating_leg = 1 - P(T₀, Tₙ)                (bond-price representation)
        fixed_leg    = K·δ · Σᵢ P(T₀, Tᵢ)           (sum of fixed-coupon PVs)

    The full discounted payoff is D(0, T₀) · V(T₀).

    Parameters
    ----------
    underlying:
        Asset identifier for the interest-rate process.
    notional:
        Notional N of the swap.
    maturity_option:
        Swaption expiry in years (T₀).
    fixed_rate:
        Fixed swap rate K (annualised).
    freq:
        Number of fixed-leg payment periods per year (e.g. 2 for semi-annual).
    swap_maturity:
        Length of the underlying swap in years.
    """

    def __init__(self, underlying: str, notional: float,
                 maturity_option: float, fixed_rate: float,
                 freq: float, swap_maturity: float):
        self._underlying    = underlying
        self._maturity      = maturity_option
        self._fixed_rate    = fixed_rate
        self._freq          = freq
        self._swap_maturity = swap_maturity
        self._notional      = notional

    @property
    def underlyings(self) -> list[str]:
        return [self._underlying]

    @property
    def maturity(self) -> float:
        return self._maturity

    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        """Compute the discounted swaption payoff across all paths.

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
        T0      = self._maturity
        Tn      = T0 + self._swap_maturity
        delta_i = 1.0 / self._freq

        p = paths[self._underlying]

        # Floating leg value at T₀ expressed as (1 - P(T₀, Tₙ))
        floating_leg = 1.0 - p.zcp(T0, Tn)

        # Fixed leg: sum of discounted fixed coupon payments
        n         = int(self._swap_maturity * self._freq)
        fixed_leg = sum(
            self._fixed_rate * delta_i * p.zcp(T0, T0 + i * delta_i)
            for i in range(1, n + 1)
        )

        # Payer swaption: max(float - fixed, 0)
        V_t = floating_leg - fixed_leg
        return p.df() * self._notional * np.maximum(V_t, 0.0)
