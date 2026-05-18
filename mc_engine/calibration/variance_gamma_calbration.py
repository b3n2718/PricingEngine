import numpy as np
from mc_engine.calibration.base import Calibrator
from mc_engine.calibration.implied_vol import BlackScholes
from mc_engine.calibration.fourier import FourierPricer
from mc_engine.market.vol_surface import VolSurface


class VarianceGammaCalibrator(Calibrator):

    def __init__(self, S: float, r: float, q: float,
                 vol_surface: VolSurface):
        self.S           = S
        self.r           = r
        self.q           = q
        self.vol_surface = vol_surface
        self._fourier    = FourierPricer()

    def _initial_params(self) -> list:
        return [0.2, -0.1, 0.2]

    def _bounds(self) -> list:
        return [
            (1e-4, 1.0),    # vol
            (-0.5, 0.5),    # theta
            (1e-4, 2.0),    # nu
        ]

    def _parse_result(self, x: np.ndarray) -> dict:
        return {"vol": x[0], "theta": x[1], "nu": x[2]}

    def _objective(self, params: np.ndarray) -> float:
        vol, theta, nu = params

        # Martingale-Bedingung
        if 1 - theta*nu - 0.5*vol**2*nu <= 0:
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
                iv    = BlackScholes.implied_vol(
                    price, self.S, K, T, self.r, self.q
                )

                if not np.isnan(iv):
                    model_vols.append(iv)
                    market_vols.append(mv)

        if not model_vols:
            return 1e6
        return self._rmse(np.array(model_vols), np.array(market_vols))

    def _characteristic_function(self, vol, theta,
                                   nu, T) -> callable:
        S, r, q = self.S, self.r, self.q
        omega   = (1/nu) * np.log(1 - theta*nu - 0.5*vol**2*nu)

        def cf(u):
            return np.exp(
                1j*u*(np.log(S) + (r - q + omega)*T)
                - T/nu * np.log(1 - 1j*u*theta*nu + 0.5*vol**2*nu*u**2)
            )

        return cf