import numpy as np
from mc_engine.products.base import Product
from mc_engine.paths.base import PathData


class AmericanOption(Product):
    """American-style call or put option priced by Longstaff-Schwartz (LSM).

    The Longstaff-Schwartz algorithm approximates the optimal exercise boundary
    by regressing the continuation value on a polynomial basis of the spot
    price at each time step.  The algorithm works backwards from maturity:

    1. At each time step, identify in-the-money paths.
    2. Regress their discounted future payoffs on a degree-2 polynomial of S.
    3. Exercise where the intrinsic value exceeds the estimated continuation.

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
        """Run LSM backward induction and return the option value per path.

        Parameters
        ----------
        paths:
            Full price path for the underlying asset (requires ``full_price_path``).
        discount:
            Deterministic discount factor D(0, T) — used to infer the
            risk-free rate for intra-step discounting.

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]`` — value at time 0 for each path.
        """
        S  = paths[self._underlying].full_price_path()   # [n_sims, n_steps]
        V  = np.maximum(S - self.strike, 0)              # intrinsic value grid
        dt = self._maturity / S.shape[1]

        # Per-step discount factor derived from the curve discount factor
        discount_rate = -np.log(discount) / self._maturity
        dcf           = np.exp(-discount_rate * dt)

        # Backward induction: step from T-1 down to 0
        for i in range(S.shape[1] - 1):
            t      = -(i + 2)      # current step (negative index)
            t_next = -(i + 1)      # next step

            S_t    = S[:, t]
            V_next = V[:, t_next]

            itm = V_next > 0       # in-the-money paths for regression basis
            if np.sum(itm) == 0:
                continue

            # Regress continuation value on degree-2 polynomial of spot
            Y     = V_next[itm] * dcf
            coeff = np.polyfit(S_t[itm], Y, 2)
            C     = np.polyval(coeff, S_t)   # estimated continuation for all paths

            E = np.maximum(S_t - self.strike, 0)  # intrinsic (exercise) value

            # Update value: exercise where intrinsic > continuation
            V[:, t] = np.where(E > C, E, V_next * dcf)

        return V[:, 0]
