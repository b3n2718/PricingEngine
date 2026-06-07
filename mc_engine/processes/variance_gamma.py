from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import EquityMarketData


class VarianceGammaProcess(StochasticProcess):
    """Variance-Gamma (VG) jump-diffusion process.

    The log-price is represented as Brownian motion with drift evaluated at a
    random Gamma time-change:

        X(t) = θ Γ(t; 1, ν) + σ W(Γ(t; 1, ν))

    where Γ(t; 1, ν) is a Gamma process with mean t and variance νt.  The
    martingale correction ω = (1/ν) ln(1 - θν - 0.5σ²ν) ensures the
    discounted price is a martingale.

    The model captures skewness (via θ) and excess kurtosis (via ν) observed
    in equity markets.

    Parameters
    ----------
    mkt:
        Equity market data (spot, dividend yield, yield curve).
    vol:
        Volatility σ of the Brownian component.
    theta:
        Drift θ of the Brownian motion in the Gamma time-change (controls
        skewness).
    nu:
        Variance ν of the Gamma time-change (controls kurtosis).
    """

    process_type = "VARIANCEGAMMA"
    noise_dim    = 1

    def __init__(self, mkt: EquityMarketData,
                 vol: float, theta: float, nu: float):
        self.mkt   = mkt
        self.vol   = vol
        self.theta = theta
        self.nu    = nu
        # Normal draw for W and a Gamma draw for the time-change (updated each run)
        self.noise = [
            {"type": "normal", "mu": 0, "sigma": 1},
            {"type": "gamma", "a": 1, "b": 1},
        ]

    def to_cpp_params(self) -> dict:
        return {
            "type":           "VARIANCEGAMMA",
            "spot":           self.mkt.spot,
            "vol":            self.vol,
            "risk_free_rate": self.mkt.curve.forward_rate(0.0),
            "div_yield":      self.mkt.div_yield,
            "theta":          self.theta,
            "nu":             self.nu,
            "path_type":      "spot",
        }

    def set_parameters(self, params: dict) -> None:
        # Gamma shape a = dt/ν and scale b = ν must be updated each run so the
        # increments Γ(dt; 1, ν) have the correct mean dt and variance νdt.
        self.a     = params["dt"] / self.nu
        self.b     = self.nu
        self.noise = [
            {"type": "normal", "mu": 0, "sigma": 1},
            {"type": "gamma",  "a": self.a, "b": self.b},
        ]
