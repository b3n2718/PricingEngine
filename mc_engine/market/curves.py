import numpy as np

class YieldCurve:
    def __init__(self, tenors: np.ndarray, rates: np.ndarray):
        self.tenors = tenors
        self.rates  = rates

    def discount(self, t: float) -> float:
        r = float(np.interp(t, self.tenors, self.rates))
        return np.exp(-r * t)

    def forward_rate(self, t: float, dt: float = 1e-6) -> float:
        """Instantaner Forward-Zinssatz bei t."""
        r1 = float(np.interp(t,      self.tenors, self.rates))
        r2 = float(np.interp(t + dt, self.tenors, self.rates))
        return (r2 * (t + dt) - r1 * t) / dt