from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import EquityMarketData

class HESTONProcess(StochasticProcess):
    process_type = "HESTON"
    noise_dim    = 2

    def __init__(self, mkt: EquityMarketData,v0: float,kappa: float, theta: float, xi: float, rho: float):
        self.mkt = mkt
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.noise = [{"type":"normal","mu":0,"sigma":1},
                      {"type":"normal","mu":0,"sigma":1}]

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
            "path_type":      "spot"
        }

    def set_parameters(self,params:dict) -> None:
        pass