import numpy as np
from scipy.stats import norm, ncx2
from mc_engine.calibration.base import Calibrator


class ShortRateCalibrator(Calibrator):
    """Abstract maximum-likelihood calibrator for short-rate time series.

    Subclasses implement ``_transition_logpdf`` for their specific transition
    distribution.  The objective is the negative log-likelihood of the observed
    rate series, which is minimised by L-BFGS-B.

    Parameters
    ----------
    rates:
        Observed historical short rates as a 1-D array.
    dt:
        Time step between consecutive observations (in years).
    """

    def __init__(self, rates: np.ndarray, dt: float):
        self.rates = rates
        self.dt    = dt

    def _initial_params(self) -> list:
        # kappa, theta (mean rate), vol
        return [0.1, float(self.rates.mean()), 0.01]

    def _bounds(self) -> list:
        return [(1e-4, 5.0), (1e-4, 0.2), (1e-4, 0.5)]

    def _parse_result(self, x: np.ndarray) -> dict:
        return {"kappa": x[0], "theta": x[1], "vol": x[2]}

    def _objective(self, params: np.ndarray) -> float:
        """Negative log-likelihood of the transition densities."""
        kappa, theta, vol = params
        ll = np.sum(self._transition_logpdf(
            self.rates[1:], self.rates[:-1], kappa, theta, vol
        ))
        return -ll

    def _transition_logpdf(self, r_next: np.ndarray, r_curr: np.ndarray,
                           kappa: float, theta: float,
                           vol: float) -> np.ndarray:
        raise NotImplementedError


class VasicekCalibrator(ShortRateCalibrator):
    """Exact MLE calibrator for the Vasicek model.

    The Vasicek model has Gaussian transitions:

        r_{t+dt} | r_t  ~  N(μ, σ²)

    where:
        μ  = r_t · exp(-κ dt) + θ · (1 - exp(-κ dt))
        σ² = (σ_vol² / 2κ) · (1 - exp(-2κ dt))
    """

    def _transition_logpdf(self, r_next: np.ndarray, r_curr: np.ndarray,
                           kappa: float, theta: float,
                           vol: float) -> np.ndarray:
        mu  = r_curr * np.exp(-kappa * self.dt) + theta * (1 - np.exp(-kappa * self.dt))
        var = vol**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * self.dt))
        return norm.logpdf(r_next, loc=mu, scale=np.sqrt(var))


class CIRCalibrator(ShortRateCalibrator):
    """Exact MLE calibrator for the Cox-Ingersoll-Ross model.

    The CIR model has non-central chi-squared transitions.  The Feller
    condition 2κθ > σ² is enforced as a hard constraint so the rate
    process stays strictly positive.
    """

    def _objective(self, params: np.ndarray) -> float:
        kappa, theta, vol = params
        # Feller condition: 2κθ > σ² guarantees strictly positive rates
        if 2 * kappa * theta <= vol**2:
            return 1e10
        return super()._objective(params)

    def _transition_logpdf(self, r_next: np.ndarray, r_curr: np.ndarray,
                           kappa: float, theta: float,
                           vol: float) -> np.ndarray:
        dt = self.dt
        c  = 2 * kappa / (vol**2 * (1 - np.exp(-kappa * dt)))
        df = 4 * kappa * theta / vol**2          # degrees of freedom
        nc = 2 * c * r_curr * np.exp(-kappa * dt)  # non-centrality parameter

        # Scale observed values and evaluate the ncx2 log-density
        obs = 2 * c * r_next
        return ncx2.logpdf(obs, df=df, nc=nc) + np.log(2 * c)
