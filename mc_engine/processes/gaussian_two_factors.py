import numpy as np
from mc_engine.processes.base import StochasticProcess
from mc_engine.market.curves import YieldCurve


class G2ppProcess(StochasticProcess):
    """G2++ two-factor Hull-White short-rate model.

    The short rate is decomposed into two mean-reverting Gaussian factors plus
    a deterministic shift that fits the initial yield curve exactly:

        r(t) = x(t) + y(t) + φ(t)
        dx = -a x dt + σ dW₁
        dy = -b y dt + η dW₂
        dW₁ dW₂ = ρ dt

    The shift φ(t) is computed analytically in the path container (G2ppPath)
    and guarantees that simulated discount factors reproduce the observed curve.

    Zero-coupon bond prices are available in closed form, making the model
    particularly efficient for fixed-income derivatives that require many bond
    evaluations (e.g. swaptions).

    Parameters
    ----------
    curve:
        Initial yield curve used for the deterministic shift φ(t).
    a:
        Mean-reversion speed of the first factor x(t).
    b:
        Mean-reversion speed of the second factor y(t).
    sigma:
        Volatility σ of the first factor.
    eta:
        Volatility η of the second factor.
    rho:
        Instantaneous correlation ρ between the two Brownian drivers.
    x0:
        Initial value of the first factor (default 0).
    y0:
        Initial value of the second factor (default 0).
    """

    process_type = "G2PP"
    noise_dim    = 2

    def __init__(self, curve: YieldCurve,
                 a: float, b: float,
                 sigma: float, eta: float,
                 rho: float,
                 x0: float = 0.0, y0: float = 0.0):
        self.curve = curve
        self.a     = a
        self.b     = b
        self.sigma = sigma
        self.eta   = eta
        self.rho   = rho
        self.x0    = x0
        self.y0    = y0
        self.noise = [
            {"type": "normal", "mu": 0, "sigma": 1},  # driver for x
            {"type": "normal", "mu": 0, "sigma": 1},  # driver for y
        ]

    def to_cpp_params(self) -> dict:
        return {
            "type":         "G2PP",
            "a":            self.a,
            "b":            self.b,
            "sigma":        self.sigma,
            "eta":          self.eta,
            "rho":          self.rho,
            "x0":           self.x0,
            "y0":           self.y0,
            "curve_tenors": self.curve.tenors.tolist(),
            "curve_rates":  self.curve.rates.tolist(),
            "path_type":    "spot",
        }

    def set_parameters(self, params: dict) -> None:
        pass
