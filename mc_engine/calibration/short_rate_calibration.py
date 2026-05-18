import numpy as np
from scipy.stats import norm, ncx2
from mc_engine.calibration.base import Calibrator


class ShortRateCalibrator(Calibrator):

    def __init__(self, rates: np.ndarray, dt: float):
        self.rates = rates
        self.dt    = dt

    def _initial_params(self) -> list:
        return [0.1, float(self.rates.mean()), 0.01]

    def _bounds(self) -> list:
        return [(1e-4, 5.0), (1e-4, 0.2), (1e-4, 0.5)]

    def _parse_result(self, x: np.ndarray) -> dict:
        return {"kappa": x[0], "theta": x[1], "vol": x[2]}

    def _objective(self, params: np.ndarray) -> float:
        kappa, theta, vol = params
        ll = np.sum(self._transition_logpdf(
            self.rates[1:], self.rates[:-1], kappa, theta, vol
        ))
        return -ll

    def _transition_logpdf(self, r_next: np.ndarray,
                           r_curr: np.ndarray,
                           kappa: float, theta: float,
                           vol: float) -> np.ndarray:
        raise NotImplementedError


class VasicekCalibrator(ShortRateCalibrator):
    """
    Exakte MLE für Vasicek.
    Übergänge sind normalverteilt:
    r_{t+dt} | r_t ~ N(mu, sigma²)
    """

    def _transition_logpdf(self, r_next, r_curr,
                           kappa, theta, vol) -> np.ndarray:
        mu  = (r_curr * np.exp(-kappa * self.dt)
               + theta * (1 - np.exp(-kappa * self.dt)))
        var = vol**2 / (2*kappa) * (1 - np.exp(-2*kappa*self.dt))
        return norm.logpdf(r_next, loc=mu, scale=np.sqrt(var))


class CIRCalibrator(ShortRateCalibrator):
    """
    Exakte MLE für CIR.
    Übergänge folgen nicht-zentraler Chi-Quadrat Verteilung.
    """

    def _objective(self, params: np.ndarray) -> float:
        kappa, theta, vol = params
        # Feller-Bedingung
        if 2 * kappa * theta <= vol**2:
            return 1e10
        return super()._objective(params)

    def _transition_logpdf(self, r_next, r_curr,
                           kappa, theta, vol) -> np.ndarray:
        dt = self.dt
        c  = 2*kappa / (vol**2 * (1 - np.exp(-kappa*dt)))
        df = 4*kappa*theta / vol**2
        nc = 2 * c * r_curr * np.exp(-kappa*dt)

        # Skalierung für ncx2
        obs = 2 * c * r_next
        return ncx2.logpdf(obs, df=df, nc=nc) + np.log(2*c)