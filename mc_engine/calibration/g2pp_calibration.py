import numpy as np
from mc_engine.calibration.base import Calibrator
from mc_engine.market.curves import YieldCurve


class G2ppCalibrator(Calibrator):
    """
    Kalibriert G2++ auf ATM Swaption-Volatilitäten.
    """

    def __init__(self, curve: YieldCurve,
                 swaption_expiries: np.ndarray,
                 swaption_tenors:   np.ndarray,
                 market_vols:       np.ndarray,
                 freq: float = 0.5):
        self.curve     = curve
        self.expiries  = swaption_expiries   # [n_expiries]
        self.tenors    = swaption_tenors     # [n_tenors]
        self.mkt_vols  = market_vols         # [n_expiries, n_tenors]
        self.freq      = freq

    def _initial_params(self) -> list:
        return [0.1, 0.3, 0.01, 0.01, -0.3]

    def _bounds(self) -> list:
        return [
            (1e-4, 2.0),    # a
            (1e-4, 2.0),    # b
            (1e-4, 0.2),    # sigma
            (1e-4, 0.2),    # eta
            (-0.99, 0.99),  # rho
        ]

    def _parse_result(self, x: np.ndarray) -> dict:
        return {
            "a": x[0], "b": x[1],
            "sigma": x[2], "eta": x[3], "rho": x[4]
        }

    def _objective(self, params: np.ndarray) -> float:
        a, b, sigma, eta, rho = params

        model_vols  = []
        market_vols = []

        for i, expiry in enumerate(self.expiries):
            for j, tenor in enumerate(self.tenors):
                mv = self.mkt_vols[i, j]
                if np.isnan(mv):
                    continue

                try:
                    iv = self._swaption_vol(
                        a, b, sigma, eta, rho, expiry, tenor
                    )
                    if not np.isnan(iv):
                        model_vols.append(iv)
                        market_vols.append(mv)
                except Exception:
                    return 1e6

        if not model_vols:
            return 1e6
        return self._rmse(np.array(model_vols), np.array(market_vols))

    def _swaption_vol(self, a, b, sigma, eta, rho,
                      expiry, tenor) -> float:
        pay_times = np.arange(
            expiry + self.freq,
            expiry + tenor + self.freq,
            self.freq
        )

        annuity = sum(
            self.freq * self.curve.discount(t) for t in pay_times
        )
        S_atm = (
            self.curve.discount(expiry) -
            self.curve.discount(expiry + tenor)
        ) / annuity

        var = self._swap_rate_variance(
            a, b, sigma, eta, rho, expiry, pay_times, annuity
        )

        # Black-Formel: σ_impl = sqrt(var / T)
        return np.sqrt(max(var, 0) / expiry)

    def _swap_rate_variance(self, a, b, sigma, eta, rho,
                             expiry, pay_times, annuity) -> float:

        def B(mean_rev, t, T):
            return (1 - np.exp(-mean_rev * (T - t))) / mean_rev

        weights = np.array([
            self.freq * self.curve.discount(t) / annuity
            for t in pay_times
        ])

        var = 0.0
        for i, Ti in enumerate(pay_times):
            for j, Tj in enumerate(pay_times):
                Bai = B(a, expiry, Ti)
                Baj = B(a, expiry, Tj)
                Bbi = B(b, expiry, Ti)
                Bbj = B(b, expiry, Tj)

                cov = (
                    sigma**2 / a**2 *
                    (1 - np.exp(-2*a*expiry)) / (2*a) * Bai * Baj
                    + eta**2 / b**2 *
                    (1 - np.exp(-2*b*expiry)) / (2*b) * Bbi * Bbj
                    + rho * sigma * eta / (a * b) * (
                        (1 - np.exp(-(a+b)*expiry)) / (a+b)
                        * (Bai*Bbj + Baj*Bbi)
                    )
                )

                var += weights[i] * weights[j] * cov

        return var