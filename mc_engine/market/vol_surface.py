import numpy as np
from scipy.interpolate import RectBivariateSpline
from mc_engine.calibration.implied_vol import BlackScholes


class VolSurface:
    """Implied volatility surface interpolated by a bicubic spline.

    Internally the surface is represented as a ``RectBivariateSpline`` over
    maturity and strike.  Inputs are automatically sorted so the spline always
    receives monotonically increasing axes.

    Parameters
    ----------
    strikes:
        1-D array of option strikes (length n_strikes).
    maturities:
        1-D array of option maturities in years (length n_maturities).
    vols:
        2-D array of implied volatilities with shape
        ``[n_maturities, n_strikes]``.
    """

    def __init__(self, strikes: np.ndarray,
                 maturities: np.ndarray,
                 vols: np.ndarray):
        # Ensure axes are sorted ascending for the spline
        sort_K = np.argsort(strikes)
        sort_T = np.argsort(maturities)

        self.strikes    = strikes[sort_K]
        self.maturities = maturities[sort_T]
        self.vols       = vols[np.ix_(sort_T, sort_K)]

        self._spline = RectBivariateSpline(
            maturities, strikes, vols, kx=3, ky=3
        )

    def implied_vol(self, T: float, K: float) -> float:
        """Interpolated implied volatility at an arbitrary (T, K) point.

        Returns a minimum of 1e-6 to avoid numerical issues in downstream
        pricing functions.

        Parameters
        ----------
        T:
            Option maturity in years.
        K:
            Option strike.

        Returns
        -------
        float
            Implied volatility (annualised).
        """
        return float(np.maximum(self._spline(T, K)[0, 0], 1e-6))

    def atm_vol(self, T: float, S: float) -> float:
        """At-the-money implied volatility (strike = spot).

        Parameters
        ----------
        T:
            Option maturity in years.
        S:
            Current spot price (used as the ATM strike).

        Returns
        -------
        float
        """
        return self.implied_vol(T, S)

    def smile(self, T: float) -> np.ndarray:
        """Volatility smile across all strikes for a given maturity.

        Parameters
        ----------
        T:
            Option maturity in years.

        Returns
        -------
        np.ndarray
            Shape ``[n_strikes]``.
        """
        return self._spline(T, self.strikes)[0]

    def total_variance(self, T: float, K: float) -> float:
        """Total variance w(T, K) = σ²(T, K) · T.

        Used in no-arbitrage checks (the surface is free of calendar arbitrage
        when w is non-decreasing in T for all K).

        Parameters
        ----------
        T:
            Option maturity in years.
        K:
            Option strike.

        Returns
        -------
        float
        """
        return self.implied_vol(T, K)**2 * T

    @classmethod
    def from_prices(cls, S: float, r: float,
                    strikes: np.ndarray,
                    maturities: np.ndarray,
                    prices: np.ndarray,
                    q: float = 0.0,
                    is_call: bool = True) -> "VolSurface":
        """Construct a VolSurface from observed option prices.

        Inverts each price to an implied volatility via the Black-Scholes
        formula using the Brent root-finding method.

        Parameters
        ----------
        S:
            Current spot price.
        r:
            Continuously-compounded risk-free rate.
        strikes:
            1-D array of strikes (length n_strikes).
        maturities:
            1-D array of maturities in years (length n_maturities).
        prices:
            2-D array of option prices with shape ``[n_maturities, n_strikes]``.
        q:
            Continuous dividend yield (default 0).
        is_call:
            True for call options (default), False for puts.

        Returns
        -------
        VolSurface
        """
        vols = np.zeros_like(prices)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                vols[i, j] = BlackScholes.implied_vol(
                    prices[i, j], S, K, T, r, q, is_call
                )
        return cls(strikes, maturities, vols)
