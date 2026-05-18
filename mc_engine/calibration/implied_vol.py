import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class BlackScholes:

    @staticmethod
    def price(S: float, K: float, T: float, r: float,
              vol: float, q: float = 0.0,
              is_call: bool = True) -> float:
        if T <= 0 or vol <= 0:
            return max(S - K, 0) if is_call else max(K - S, 0)

        d1 = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)

        if is_call:
            return (S * np.exp(-q*T) * norm.cdf(d1)
                    - K * np.exp(-r*T) * norm.cdf(d2))
        return (K * np.exp(-r*T) * norm.cdf(-d2)
                - S * np.exp(-q*T) * norm.cdf(-d1))

    @staticmethod
    def implied_vol(market_price: float, S: float, K: float,
                    T: float, r: float, q: float = 0.0,
                    is_call: bool = True) -> float:
        """
        BS implizite Volatilität via Brent-Methode.
        Gibt np.nan zurück wenn kein Wert gefunden wird.
        """
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        if market_price <= intrinsic:
            return np.nan

        try:
            return brentq(
                lambda v: BlackScholes.price(S, K, T, r, v, q, is_call)
                          - market_price,
                1e-6, 10.0,
                xtol = 1e-8,
                maxiter = 100
            )
        except ValueError:
            return np.nan

    @staticmethod
    def delta(S: float, K: float, T: float, r: float,
              vol: float, q: float = 0.0,
              is_call: bool = True) -> float:
        d1 = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        if is_call:
            return np.exp(-q*T) * norm.cdf(d1)
        return -np.exp(-q*T) * norm.cdf(-d1)

    @staticmethod
    def vega(S: float, K: float, T: float, r: float,
             vol: float, q: float = 0.0) -> float:
        d1 = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)