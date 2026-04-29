from processes.base import StochasticProcess
from market.market_data import EquityMarketData

class GBMProcess(StochasticProcess):
    process_type = "GBM"
    noise_dim    = 1

    def __init__(self, mkt: EquityMarketData, vol: float):
        self.mkt = mkt
        self.vol = vol
        self.noise = [{"type":"normal","mu":0,"sigma":1}]


    def to_cpp_params(self) -> dict:
        return {
            "type":           "GBM",
            "spot":           self.mkt.spot,
            "vol":            self.vol,
            "risk_free_rate": self.mkt.curve.forward_rate(0.0),
            "div_yield":      self.mkt.div_yield,
        }
    
    def set_parameters(self,*params) -> None:
        pass