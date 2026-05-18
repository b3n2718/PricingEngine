import numpy as np
from scipy.interpolate import RectBivariateSpline
from mc_engine.calibration.implied_vol import BlackScholes


class VolSurface:
    """
    Implizite Volatilitätsoberfläche über Strike und Laufzeit.
    Intern als kubischer Spline interpoliert.
    """

    def __init__(self, strikes: np.ndarray,
                 maturities: np.ndarray,
                 vols: np.ndarray):
        """
        strikes:    [n_strikes]
        maturities: [n_maturities]
        vols:       [n_maturities, n_strikes]
        """
        
        sort_K = np.argsort(strikes)
        sort_T = np.argsort(maturities)

        self.strikes    = strikes[sort_K]
        self.maturities = maturities[sort_T]
        self.vols       = vols[np.ix_(sort_T, sort_K)]
        
        self._spline    = RectBivariateSpline(
            maturities, strikes, vols, kx=3, ky=3
        )

    def implied_vol(self, T: float, K: float) -> float:
        """Interpolierte implizite Vol für beliebiges (T, K)."""
        return float(np.maximum(self._spline(T, K)[0, 0], 1e-6))

    def atm_vol(self, T: float, S: float) -> float:
        """ATM Vol — Strike = Spot."""
        return self.implied_vol(T, S)

    def smile(self, T: float) -> np.ndarray:
        """Vol Smile für eine Laufzeit — [n_strikes]."""
        return self._spline(T, self.strikes)[0]

    def total_variance(self, T: float, K: float) -> float:
        """Totale Varianz w(T,K) = σ²(T,K) * T — für Arbitrage-Check."""
        return self.implied_vol(T, K)**2 * T

    @classmethod
    def from_prices(cls, S: float, r: float,
                    strikes: np.ndarray,
                    maturities: np.ndarray,
                    prices: np.ndarray,
                    q: float = 0.0,
                    is_call: bool = True) -> "VolSurface":
        """
        Konstruiert Vol Surface direkt aus Optionspreisen.
        prices: [n_maturities, n_strikes]
        """
        vols = np.zeros_like(prices)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                vols[i, j] = BlackScholes.implied_vol(
                    prices[i, j], S, K, T, r, q, is_call
                )
        return cls(strikes, maturities, vols)