import numpy as np
from mc_engine.paths.base import PathData
from mc_engine.market.curves import YieldCurve


class G2ppPath(PathData):
    """Path container for the G2++ two-factor Hull-White model.

    Stores the two latent factors x(t) and y(t) with shape
    ``[n_sims, 2, n_steps]``.  The short rate is reconstructed as

        r(t) = x(t) + y(t) + φ(t)

    where φ(t) is the deterministic shift that fits the initial yield curve.

    Zero-coupon bond prices are computed analytically (no nested simulation),
    which makes pricing fixed-income derivatives very efficient.

    Parameters
    ----------
    data:
        Simulated factor paths with shape ``[n_sims, 2, n_steps]``.
        Axis 1 index 0 is x(t), index 1 is y(t).
    params:
        Parameter dict from ``G2ppProcess.to_cpp_params()`` augmented with
        ``"dt"``.
    """

    def __init__(self, data: np.ndarray, params: dict):
        self._data   = data
        self.params  = params
        self._dt     = params["dt"]
        self._tenors = np.array(params["curve_tenors"])
        self._rates  = np.array(params["curve_rates"])

    # ── Internal helpers ───────────────────────────────────────────────────

    def _discount(self, t: float) -> float:
        """Interpolated initial market discount factor P(0, t)."""
        r = np.interp(t, self._tenors, self._rates)
        return np.exp(-r * t)

    @property
    def n_sims(self) -> int:
        """Number of Monte Carlo simulation paths."""
        return self._data.shape[0]

    # ── PathData interface ─────────────────────────────────────────────────

    def terminal_value(self) -> np.ndarray:
        """Short rate r(T) = x(T) + y(T) + φ(T) — shape ``[n_sims]``."""
        return self._short_rate_at(-1)

    def at_step(self, step: int) -> np.ndarray:
        """Short rate at a given time-step index — shape ``[n_sims]``."""
        return self._short_rate_at(step)

    def _short_rate_at(self, step: int) -> np.ndarray:
        t   = step * self._dt
        x_t = self._data[:, 0, step]
        y_t = self._data[:, 1, step]
        return x_t + y_t + self._phi(t)

    # ── G2++-specific bond pricing ─────────────────────────────────────────

    def zcp(self, t1: float, t2: float) -> np.ndarray:
        """Closed-form zero-coupon bond price P(t1, t2).

        The affine formula is:

            P(t1,t2) = P(0,t2)/P(0,t1) · exp(-B(a)·x(t1) - B(b)·y(t1) - ½V)

        where B(α) = (1 - exp(-α·τ)) / α and V is the variance term.

        Parameters
        ----------
        t1:
            Valuation time (years).
        t2:
            Bond maturity (years).

        Returns
        -------
        np.ndarray
            Bond prices across all paths — shape ``[n_sims]``.
        """
        t1_idx = min(int(round(t1 / self._dt)), self._data.shape[2] - 1)
        x_t1   = self._data[:, 0, t1_idx]
        y_t1   = self._data[:, 1, t1_idx]

        Ba   = self._B(self.params["a"], t1, t2)
        Bb   = self._B(self.params["b"], t1, t2)
        V    = self._V(t1, t2)
        P0t2 = self._discount(t2)
        P0t1 = self._discount(t1)

        return (P0t2 / P0t1) * np.exp(-Ba * x_t1 - Bb * y_t1 - 0.5 * V)

    def _B(self, mean_rev: float, t: float, T: float) -> float:
        """Auxiliary function B(α, t, T) = (1 - exp(-α(T-t))) / α."""
        return (1 - np.exp(-mean_rev * (T - t))) / mean_rev

    def _V(self, t: float, T: float) -> float:
        """Variance term V(t, T) entering the bond price formula.

        Accounts for the variance and covariance of x and y over [t, T].
        """
        a, b      = self.params["a"], self.params["b"]
        s, e, rho = self.params["sigma"], self.params["eta"], self.params["rho"]
        tau       = T - t

        Ba = self._B(a, t, T)
        Bb = self._B(b, t, T)

        V_x  = (s**2 / a**2) * (tau - 2*Ba + (1 - np.exp(-2*a*tau)) / (2*a))
        V_y  = (e**2 / b**2) * (tau - 2*Bb + (1 - np.exp(-2*b*tau)) / (2*b))
        V_xy = (2*rho*s*e / (a*b)) * (tau - Ba - Bb + (1 - np.exp(-(a+b)*tau)) / (a+b))

        return V_x + V_y + V_xy

    def _phi(self, t: float) -> float:
        """Deterministic shift φ(t) that matches the initial yield curve.

        φ(t) = f(0, t) + (σ²/2a²)(1-e^{-at})² + (η²/2b²)(1-e^{-bt})²
                        + (ρση/ab)(1-e^{-at})(1-e^{-bt})

        Only required for computing the short rate; not used in the analytic
        bond price formula.
        """
        a, b      = self.params["a"], self.params["b"]
        s, e, rho = self.params["sigma"], self.params["eta"], self.params["rho"]

        f0t    = self._discount(t)  # approximation: using P(0,t) for f(0,t)
        term_x  = (s**2 / (2*a**2)) * (1 - np.exp(-a*t))**2
        term_y  = (e**2 / (2*b**2)) * (1 - np.exp(-b*t))**2
        term_xy = (rho*s*e / (a*b)) * (1 - np.exp(-a*t)) * (1 - np.exp(-b*t))

        return f0t + term_x + term_y + term_xy

    def df(self) -> np.ndarray:
        """Discount factor from time 0 to the simulation horizon.

        Integrates r(t) = x(t) + y(t) + φ(t) with an Euler sum:

            D(0, T) = exp(-Σ r(tₖ) · dt)

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        x  = self._data[:, 0, :]   # [n_sims, n_steps]
        y  = self._data[:, 1, :]   # [n_sims, n_steps]

        dt      = self.params["dt"]
        n_steps = self._data.shape[2]
        times   = np.arange(n_steps) * dt
        phi     = np.array([self._phi(t) for t in times])  # [n_steps]

        r = x + y + phi[np.newaxis, :]   # [n_sims, n_steps]
        return np.exp(-r.sum(axis=1) * dt)
