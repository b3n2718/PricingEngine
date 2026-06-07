import numpy as np
from scipy.integrate import quad


class FourierPricer:
    """Option pricer via the characteristic function (Gil-Pelaez inversion).

    Applicable to any model that provides the characteristic function of
    log(S_T) under the risk-neutral measure.  The call price is computed as:

        C = S · e^{-qT} · P₁ - K · e^{-rT} · P₂

    where P₁ and P₂ are risk-neutral probabilities obtained by numerical
    integration of the Gil-Pelaez formula.  Put prices follow from put-call
    parity.

    Parameters
    ----------
    upper_bound:
        Upper integration limit for the quadrature (default 500).
    limit:
        Maximum number of quadrature sub-intervals (default 200).
    """

    def __init__(self, upper_bound: float = 500.0, limit: int = 200):
        self.upper_bound = upper_bound
        self.limit       = limit

    def call_price(self, cf: callable, S: float, K: float,
                   T: float, r: float, q: float = 0.0) -> float:
        """Price a European call option via the Gil-Pelaez formula.

        Parameters
        ----------
        cf:
            Characteristic function φ(u) of log(S_T) under the risk-neutral
            measure, accepting a complex argument u.
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

        Returns
        -------
        float
            Call price.
        """
        log_K = np.log(K)

        def integrand_P1(u):
            return np.real(
                np.exp(-1j * u * log_K) * cf(u - 1j) / (1j * u * cf(-1j))
            )

        def integrand_P2(u):
            return np.real(
                np.exp(-1j * u * log_K) * cf(u) / (1j * u)
            )

        P1 = 0.5 + (1 / np.pi) * quad(
            integrand_P1, 1e-6, self.upper_bound, limit=self.limit
        )[0]

        P2 = 0.5 + (1 / np.pi) * quad(
            integrand_P2, 1e-6, self.upper_bound, limit=self.limit
        )[0]

        return S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    def put_price(self, cf: callable, S: float, K: float,
                  T: float, r: float, q: float = 0.0) -> float:
        """Price a European put via put-call parity.

        Parameters
        ----------
        cf:
            Characteristic function of log(S_T).
        S, K, T, r, q:
            Standard option inputs.

        Returns
        -------
        float
            Put price.
        """
        call = self.call_price(cf, S, K, T, r, q)
        return call - S * np.exp(-q * T) + K * np.exp(-r * T)
