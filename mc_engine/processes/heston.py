from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import EquityMarketData


class HESTONProcess(StochasticProcess):
    """Heston stochastic-volatility model.

    The joint dynamics of the spot price S and variance v under the
    risk-neutral measure are:

        dS = (r - q) S dt + √v S dW₁
        dv = κ(θ - v) dt + ξ √v dW₂
        dW₁ dW₂ = ρ dt

    The Feller condition 2κθ > ξ² ensures the variance process stays strictly
    positive.  The calibrator enforces this constraint during optimisation.

    Parameters
    ----------
    mkt:
        Equity market data (spot, dividend yield, yield curve).
    v0:
        Initial variance (≈ σ₀²).
    kappa:
        Mean-reversion speed κ of the variance process.
    theta:
        Long-run variance θ.
    xi:
        Volatility of variance (vol-of-vol) ξ.
    rho:
        Correlation ρ between the two Brownian drivers.
    """

    process_type = "HESTON"
    noise_dim    = 2  # two correlated Brownian drivers

    def __init__(self, mkt: EquityMarketData,
                 v0: float, kappa: float, theta: float,
                 xi: float, rho: float):
        self.mkt   = mkt
        self.v0    = v0
        self.kappa = kappa
        self.theta = theta
        self.xi    = xi
        self.rho   = rho
        self.noise = [
            {"type": "normal", "mu": 0, "sigma": 1},  # spot driver
            {"type": "normal", "mu": 0, "sigma": 1},  # variance driver
        ]

    def to_cpp_params(self) -> dict:
        return {
            "type":           "HESTON",
            "spot":           self.mkt.spot,
            "v0":             self.v0,
            "kappa":          self.kappa,
            "theta":          self.theta,
            "xi":             self.xi,
            "rho":            self.rho,
            "risk_free_rate": self.mkt.curve.forward_rate(0.0),
            "div_yield":      self.mkt.div_yield,
            "path_type":      "spot",
        }

    def set_parameters(self, params: dict) -> None:
        pass  # Heston noise dimensions are independent of dt
