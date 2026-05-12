from processes.base import StochasticProcess
from market.market_data import EquityMarketData

class GammaVarianceProcess(StochasticProcess):
    process_type = "GAMMAVARIANCE"
    noise_dim    = 1

    def __init__(self, mkt: EquityMarketData, vol: float, theta: float, nu: float):
        self.mkt = mkt
        self.vol = vol
        self.theta = theta
        self.nu = nu
        self.noise = [{"type":"normal","mu":0,"sigma":1},{"type":"gamma","a":1,"b":1}]


    def to_cpp_params(self) -> dict:
        return {
            "type":           "GAMMAVARIANCE",
            "spot":           self.mkt.spot,
            "vol":            self.vol,
            "risk_free_rate": self.mkt.curve.forward_rate(0.0),
            "div_yield":      self.mkt.div_yield,
            "theta":          self.theta,
            "nu":             self.nu
        }
    
    def set_parameters(self,params:dict) -> None:
        self.a = params["dt"]/self.nu
        self.b = self.nu
        self.noise = [{"type":"normal","mu":0,"sigma":1},{"type":"gamma","a":self.a,"b":self.b}]