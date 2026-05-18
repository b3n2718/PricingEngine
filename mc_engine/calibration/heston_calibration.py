import numpy as np
from mc_engine.calibration.base import Calibrator
from mc_engine.calibration.implied_vol import BlackScholes
from mc_engine.calibration.fourier import FourierPricer
from mc_engine.market.vol_surface import VolSurface


class HestonCalibrator(Calibrator):

    def __init__(self, S: float, r: float, q: float,
                 vol_surface: VolSurface):
        self.S           = S
        self.r           = r
        self.q           = q
        self.vol_surface = vol_surface
        self._fourier    = FourierPricer()

    def _initial_params(self) -> list:
        return [0.04, 2.0, 0.04, 0.3, -0.7]

    def _bounds(self) -> list:
        return [
            (1e-4, 1.0),    # v0
            (1e-4, 10.0),   # kappa
            (1e-4, 1.0),    # theta
            (1e-4, 2.0),    # xi
            (-0.99, 0.99),  # rho
        ]

    def _parse_result(self, x: np.ndarray) -> dict:
        return {
            "v0":    x[0], "kappa": x[1],
            "theta": x[2], "xi":    x[3], "rho": x[4]
        }

    def _objective(self, params: np.ndarray) -> float:
        v0, kappa, theta, xi, rho = params

        # Feller-Bedingung
        if 2 * kappa * theta <= xi**2:
            return 1e6

        model_vols  = []
        market_vols = []

        for T in self.vol_surface.maturities:
            for K in self.vol_surface.strikes:
                mv = self.vol_surface.implied_vol(T, K)
                if np.isnan(mv):
                    continue

                cf    = self._characteristic_function(
                    v0, kappa, theta, xi, rho, T
                )
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

    def _characteristic_function(self, v0, kappa, theta,
                                   xi, rho, T) -> callable:
        S, r, q = self.S, self.r, self.q

        def cf(u):
            d = np.sqrt(
                (rho*xi*1j*u - kappa)**2 + xi**2*(1j*u + u**2)
            )
            g = (kappa - rho*xi*1j*u - d) / (kappa - rho*xi*1j*u + d)

            C = ((r - q)*1j*u*T
                 + kappa*theta/xi**2 * (
                     (kappa - rho*xi*1j*u - d)*T
                     - 2*np.log((1 - g*np.exp(-d*T)) / (1 - g))
                 ))
            D = ((kappa - rho*xi*1j*u - d) / xi**2
                 * (1 - np.exp(-d*T)) / (1 - g*np.exp(-d*T)))

            return np.exp(C + D*v0 + 1j*u*np.log(S))

        return cf