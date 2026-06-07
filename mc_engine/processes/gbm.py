from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import EquityMarketData


class GBMProcess(StochasticProcess):
    """Geometric Brownian Motion (GBM) equity process.

    Models the stock price under the risk-neutral measure as:

        dS = (r - q) S dt + σ S dW

    where r is the instantaneous risk-free rate, q is the continuous dividend
    yield, σ is the constant volatility, and W is a standard Brownian motion.

    Parameters
    ----------
    mkt:
        Equity market data containing the spot price, dividend yield and
        yield curve.
    vol:
        Constant annualised volatility σ.
    """

    process_type = "GBM"
    noise_dim    = 1  # one Brownian driver

    def __init__(self, mkt: EquityMarketData, vol: float):
        self.mkt  = mkt
        self.vol  = vol
        # One standard-normal noise dimension
        self.noise = [{"type": "normal", "mu": 0, "sigma": 1}]

    def to_cpp_params(self) -> dict:
        return {
            "type":           "GBM",
            "spot":           self.mkt.spot,
            "vol":            self.vol,
            "risk_free_rate": self.mkt.curve.forward_rate(0.0),
            "div_yield":      self.mkt.div_yield,
            "path_type":      "spot",
        }

    def set_parameters(self, params: dict) -> None:
        pass  # GBM noise dimensions are independent of dt
