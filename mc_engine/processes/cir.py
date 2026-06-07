from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import FIMarketData


class CIRProcess(StochasticProcess):
    """Cox-Ingersoll-Ross (CIR) short-rate model.

    The short rate r(t) evolves as:

        dr = κ(θ - r) dt + σ √r dW

    The square-root diffusion coefficient ensures non-negative rates when the
    Feller condition 2κθ > σ² is satisfied.  The CIRCalibrator enforces this
    constraint during optimisation.

    Parameters
    ----------
    mkt:
        Fixed-income market data containing the initial short rate.
    vol:
        Diffusion coefficient σ (volatility scaling).
    kappa:
        Mean-reversion speed κ.
    theta:
        Long-run mean θ of the short rate.
    """

    process_type = "CIR"
    noise_dim    = 1

    def __init__(self, mkt: FIMarketData,
                 vol: float, kappa: float, theta: float):
        self.mkt   = mkt
        self.vol   = vol
        self.kappa = kappa
        self.theta = theta
        self.noise = [{"type": "normal", "mu": 0, "sigma": 1}]

    def to_cpp_params(self) -> dict:
        return {
            "type":      "CIR",
            "r_spot":    self.mkt.r_spot,
            "vol":       self.vol,
            "kappa":     self.kappa,
            "theta":     self.theta,
            "path_type": "spot",
        }

    def set_parameters(self, params: dict) -> None:
        pass
