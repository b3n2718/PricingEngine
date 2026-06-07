import numpy as np


class YieldCurve:
    """Piecewise-linear yield curve backed by tenor/rate pairs.

    Rates are linearly interpolated between the provided knot points.  The
    curve is used to compute discount factors and instantaneous forward rates
    required by both pricing and calibration routines.

    Parameters
    ----------
    tenors:
        Array of time points (in years), e.g. ``[0.5, 1, 2, 5, 10]``.
    rates:
        Continuously-compounded zero rates corresponding to each tenor.
    """

    def __init__(self, tenors: np.ndarray, rates: np.ndarray):
        self.tenors = tenors
        self.rates  = rates

    def discount(self, t: float) -> float:
        """Compute the discount factor D(0, t) = exp(-r(t) · t).

        Parameters
        ----------
        t:
            Maturity in years.

        Returns
        -------
        float
            Discount factor in (0, 1].
        """
        r = float(np.interp(t, self.tenors, self.rates))
        return np.exp(-r * t)

    def forward_rate(self, t: float, dt: float = 1e-6) -> float:
        """Instantaneous forward rate f(0, t) via finite difference.

        Derived from the zero-rate curve R(t) as:

            f(0, t) = d/dt [R(t) · t] ≈ (R(t+dt)·(t+dt) - R(t)·t) / dt

        Parameters
        ----------
        t:
            Maturity in years.
        dt:
            Finite-difference step (default 1e-6).

        Returns
        -------
        float
            Instantaneous forward rate at time t.
        """
        r1 = float(np.interp(t,      self.tenors, self.rates))
        r2 = float(np.interp(t + dt, self.tenors, self.rates))
        return (r2 * (t + dt) - r1 * t) / dt
