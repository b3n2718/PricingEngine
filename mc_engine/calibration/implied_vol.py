import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class BlackScholes:
    """Black-Scholes closed-form option pricing and implied volatility tools.

    All methods are static so the class acts as a namespace.
    """

    @staticmethod
    def price(S: float, K: float, T: float, r: float,
              vol: float, q: float = 0.0,
              is_call: bool = True) -> float:
        """Black-Scholes option price.

        Returns the intrinsic value when T ≤ 0 or vol ≤ 0.

        Parameters
        ----------
        S:
            Current spot price.
        K:
            Strike price.
        T:
            Time to expiry in years.
        r:
            Continuously-compounded risk-free rate.
        vol:
            Implied (or model) volatility.
        q:
            Continuous dividend yield (default 0).
        is_call:
            True for call, False for put.

        Returns
        -------
        float
            Option price.
        """
        if T <= 0 or vol <= 0:
            return max(S - K, 0) if is_call else max(K - S, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)

        if is_call:
            return (S * np.exp(-q * T) * norm.cdf(d1)
                    - K * np.exp(-r * T) * norm.cdf(d2))
        return (K * np.exp(-r * T) * norm.cdf(-d2)
                - S * np.exp(-q * T) * norm.cdf(-d1))

    @staticmethod
    def implied_vol(market_price: float, S: float, K: float,
                    T: float, r: float, q: float = 0.0,
                    is_call: bool = True) -> float:
        """Black-Scholes implied volatility via the Brent root-finding method.

        Returns ``np.nan`` if the market price is below intrinsic value or if
        no root is found within [1e-6, 10.0].

        Parameters
        ----------
        market_price:
            Observed option price.
        S:
            Current spot price.
        K:
            Strike price.
        T:
            Time to expiry in years.
        r:
            Continuously-compounded risk-free rate.
        q:
            Continuous dividend yield (default 0).
        is_call:
            True for call (default), False for put.

        Returns
        -------
        float
            Implied volatility, or ``np.nan`` on failure.
        """
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        if market_price <= intrinsic:
            return np.nan

        try:
            return brentq(
                lambda v: BlackScholes.price(S, K, T, r, v, q, is_call) - market_price,
                1e-6, 10.0,
                xtol    = 1e-8,
                maxiter = 100,
            )
        except ValueError:
            return np.nan

    @staticmethod
    def delta(S: float, K: float, T: float, r: float,
              vol: float, q: float = 0.0,
              is_call: bool = True) -> float:
        """Black-Scholes delta (∂price / ∂S).

        Parameters
        ----------
        S, K, T, r, vol, q:
            Standard Black-Scholes inputs.
        is_call:
            True for call delta, False for put delta.

        Returns
        -------
        float
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        if is_call:
            return np.exp(-q * T) * norm.cdf(d1)
        return -np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def vega(S: float, K: float, T: float, r: float,
             vol: float, q: float = 0.0) -> float:
        """Black-Scholes vega (∂price / ∂vol), same for calls and puts.

        Parameters
        ----------
        S, K, T, r, vol, q:
            Standard Black-Scholes inputs.

        Returns
        -------
        float
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
