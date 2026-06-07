import numpy as np
from mc_engine.calibration.base import Calibrator
from mc_engine.calibration.implied_vol import BlackScholes
from mc_engine.calibration.fourier import FourierPricer
from mc_engine.market.vol_surface import VolSurface


class VarianceGammaCalibrator(Calibrator):
    """Calibrate the Variance-Gamma model to a market implied-volatility surface.

    Minimises the RMSE between VG model-implied volatilities (computed via
    Fourier inversion of the VG characteristic function) and the market surface.

    The martingale condition 1 - θν - 0.5σ²ν > 0 is enforced as a hard
    constraint to ensure the compensated process is a valid martingale.

    Parameters
    ----------
    S:
        Current spot price.
    r:
        Continuously-compounded risk-free rate.
    q:
        Continuous dividend yield.
    vol_surface:
        Market implied-volatility surface to calibrate against.
    """

    def __init__(self, S: float, r: float, q: float,
                 vol_surface: VolSurface):
        self.S           = S
        self.r           = r
        self.q           = q
        self.vol_surface = vol_surface
        self._fourier    = FourierPricer()

    def _initial_params(self) -> list:
        # vol, theta (skew), nu (kurtosis)
        return [0.2, -0.1, 0.2]

    def _bounds(self) -> list:
        return [
            (1e-4, 1.0),    # vol:   Brownian volatility σ
            (-0.5, 0.5),    # theta: drift in Gamma time-change (skewness)
            (1e-4, 2.0),    # nu:    variance of time-change (kurtosis)
        ]

    def _parse_result(self, x: np.ndarray) -> dict:
        return {"vol": x[0], "theta": x[1], "nu": x[2]}

    def _objective(self, params: np.ndarray) -> float:
        vol, theta, nu = params

        # Martingale condition: ensures the risk-neutral drift is well-defined
        if 1 - theta * nu - 0.5 * vol**2 * nu <= 0:
            return 1e6

        model_vols  = []
        market_vols = []

        for T in self.vol_surface.maturities:
            for K in self.vol_surface.strikes:
                mv = self.vol_surface.implied_vol(T, K)
                if np.isnan(mv):
                    continue

                cf    = self._characteristic_function(vol, theta, nu, T)
                price = self._fourier.call_price(cf, self.S, K, T, self.r, self.q)
                iv    = BlackScholes.implied_vol(price, self.S, K, T, self.r, self.q)

                if not np.isnan(iv):
                    model_vols.append(iv)
                    market_vols.append(mv)

        if not model_vols:
            return 1e6
        return self._rmse(np.array(model_vols), np.array(market_vols))

    def _characteristic_function(self, vol: float, theta: float,
                                   nu: float, T: float) -> callable:
        """Variance-Gamma characteristic function of log(S_T).

        The VG CF has the closed form:

            φ(u) = exp(iu·(log(S) + (r-q+ω)T) - (T/ν)·log(1 - iuθν + 0.5σ²νu²))

        where ω = (1/ν)·log(1 - θν - 0.5σ²ν) is the martingale correction.
        """
        S, r, q = self.S, self.r, self.q
        omega   = (1 / nu) * np.log(1 - theta * nu - 0.5 * vol**2 * nu)

        def cf(u):
            return np.exp(
                1j * u * (np.log(S) + (r - q + omega) * T)
                - T / nu * np.log(1 - 1j * u * theta * nu + 0.5 * vol**2 * nu * u**2)
            )

        return cf
