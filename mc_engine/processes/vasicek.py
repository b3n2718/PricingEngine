from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import FIMarketData


class VasicekProcess(StochasticProcess):
    """Vasicek mean-reverting short-rate model.

    The short rate r(t) evolves as:

        dr = κ(θ - r) dt + σ dW

    The model allows negative rates and has Gaussian transition densities,
    which enables exact maximum-likelihood calibration (see VasicekCalibrator).

    Parameters
    ----------
    mkt:
        Fixed-income market data containing the initial short rate.
    vol:
        Constant short-rate volatility σ.
    kappa:
        Mean-reversion speed κ.
    theta:
        Long-run mean θ of the short rate.
    """

    process_type = "VASICEK"
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
            "type":      "VASICEK",
            "r_spot":    self.mkt.r_spot,
            "vol":       self.vol,
            "kappa":     self.kappa,
            "theta":     self.theta,
            "path_type": "spot",
        }

    def set_parameters(self, params: dict) -> None:
        pass
