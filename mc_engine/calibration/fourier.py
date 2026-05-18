import numpy as np
from scipy.integrate import quad


class FourierPricer:
    """
    Generischer Option-Pricer via charakteristische Funktion.
    Funktioniert für jedes Modell das eine CF des Log-Preises hat.
    """

    def __init__(self, upper_bound: float = 500.0,
                 limit: int = 200):
        self.upper_bound = upper_bound
        self.limit       = limit

    def call_price(self, cf: callable, S: float, K: float,
                   T: float, r: float, q: float = 0.0) -> float:
        """
        Call-Preis via Gil-Pelaez Formel:
        C = S*e^{-qT}*P1 - K*e^{-rT}*P2

        cf: charakteristische Funktion φ(u) des Log-Preises log(S_T)
        """
        log_K = np.log(K)

        def integrand_P1(u):
            return np.real(
                np.exp(-1j * u * log_K) *
                cf(u - 1j) / (1j * u * cf(-1j))
            )

        def integrand_P2(u):
            return np.real(
                np.exp(-1j * u * log_K) * cf(u) / (1j * u)
            )

        P1 = 0.5 + (1/np.pi) * quad(
            integrand_P1, 1e-6, self.upper_bound,
            limit=self.limit
        )[0]

        P2 = 0.5 + (1/np.pi) * quad(
            integrand_P2, 1e-6, self.upper_bound,
            limit=self.limit
        )[0]

        return S * np.exp(-q*T) * P1 - K * np.exp(-r*T) * P2

    def put_price(self, cf: callable, S: float, K: float,
                  T: float, r: float, q: float = 0.0) -> float:
        """Put via Put-Call Parität."""
        call = self.call_price(cf, S, K, T, r, q)
        return call - S * np.exp(-q*T) + K * np.exp(-r*T)