import numpy as np
from mc_engine.paths.base import PathData
from mc_engine.market.curves import YieldCurve


class G2ppPath(PathData):
    """
    G2++ Pfad — state shape: [n_sims, n_steps, 2]
    Spalte 0: x(t), Spalte 1: y(t)

    Bondpreise analytisch ohne nested MC.
    """

    def __init__(self, data: np.ndarray, params: dict):
        self._data  = data
        self.params = params
        self._dt    = params["dt"]
        # Kurve aus params rekonstruieren
        self._tenors = np.array(params["curve_tenors"])
        self._rates  = np.array(params["curve_rates"])

    def _discount(self, t: float) -> float:
        r = np.interp(t, self._tenors, self._rates)
        return np.exp(-r * t)


    @property
    def n_sims(self) -> int:
        return self._data.shape[0]

    def terminal_value(self) -> np.ndarray:
        """Short-Rate am Ende — x(T) + y(T) + phi(T)"""
        return self._short_rate_at(-1)

    def at_step(self, step: int) -> np.ndarray:
        """Short-Rate zum Zeitschritt."""
        return self._short_rate_at(step)

    def _short_rate_at(self, step: int) -> np.ndarray:
        t   = self.times[step]
        x_t = self._data[:, 0, step]
        y_t = self._data[:, 1, step]
        return x_t + y_t + self._phi(t)

    def zcp(self, t1: float, t2: float) -> np.ndarray:
        """
        P(t1, t2) = P(0,t2)/P(0,t1) * exp(-B(a)*x(t1) - B(b)*y(t1) - 0.5*V)
        shape: [n_sims]
        """
        t1_idx = min(int(round(t1 / self._dt)), self._data.shape[2] - 1)
        x_t1   = self._data[:,0, t1_idx]
        y_t1   = self._data[:,1, t1_idx]

        Ba  = self._B(self.params["a"], t1, t2)
        Bb  = self._B(self.params["b"], t1, t2)
        V   = self._V(t1, t2)

        P0t2 = self._discount(t2)
        P0t1 = self._discount(t1)

        return (P0t2 / P0t1) * np.exp(-Ba * x_t1 - Bb * y_t1 - 0.5 * V)

    def _B(self, mean_rev: float, t: float, T: float) -> float:
        return (1 - np.exp(-mean_rev * (T - t))) / mean_rev

    def _V(self, t: float, T: float) -> float:
        """Varianz-Term V(t,T)."""
        a, b      = self.params["a"], self.params["b"]
        s, e, rho = self.params["sigma"], self.params["eta"], self.params["rho"]
        tau       = T - t

        Ba  = self._B(a, t, T)
        Bb  = self._B(b, t, T)

        V_x = (s**2 / a**2) * (
            tau - 2*Ba + (1 - np.exp(-2*a*tau)) / (2*a)
        )
        V_y = (e**2 / b**2) * (
            tau - 2*Bb + (1 - np.exp(-2*b*tau)) / (2*b)
        )
        V_xy = (2 * rho * s * e / (a * b)) * (
            tau - Ba - Bb
            + (1 - np.exp(-(a+b)*tau)) / (a+b)
        )

        return V_x + V_y + V_xy

    def _phi(self, t: float) -> float:
        """
        Deterministischer Shift — fittet Anfangskurve.
        Wird nur für Short-Rate benötigt, nicht für Bondpreise.
        """
        a, b      = self.params["a"], self.params["b"]
        s, e, rho = self.params["sigma"], self.params["eta"], self.params["rho"]

        f0t = self._discount(t)

        term_x  = (s**2 / (2*a**2)) * (1 - np.exp(-a*t))**2
        term_y  = (e**2 / (2*b**2)) * (1 - np.exp(-b*t))**2
        term_xy = (rho*s*e / (a*b)) * (1 - np.exp(-a*t)) * (1 - np.exp(-b*t))

        return f0t + term_x + term_y + term_xy
    
    def df(self) -> np.ndarray:
        """
        Diskontfaktor von 0 bis T_end.
        ∫_0^T r(t) dt = ∫_0^T (x(t) + y(t) + φ(t)) dt
        shape: [n_sims]
        """
        x = self._data[:, 0, :]   # [n_sims, n_steps]
        y = self._data[:, 1, :]   # [n_sims, n_steps]

        # φ(t) für jeden Zeitschritt — deterministisch, einmal berechnen
        dt     = self.params["dt"]
        n_steps = self._data.shape[2]
        times  = np.arange(n_steps) * dt
        phi    = np.array([self._phi(t) for t in times])   # [n_steps]

        # r(t) = x(t) + y(t) + φ(t)
        r = x + y + phi[np.newaxis, :]   # [n_sims, n_steps]

        return np.exp(-r.sum(axis=1) * dt)