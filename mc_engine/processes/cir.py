from processes.base import StochasticProcess
from market.market_data import FISpotRateMarketData

class CIRProcess(StochasticProcess):
    process_type = "CIR"
    noise_dim    = 1

    def __init__(self, mkt: FISpotRateMarketData, vol: float, kappa: float, theta: float):
        self.mkt = mkt
        self.vol = vol
        self.kappa = kappa
        self.theta = theta
        self.noise = [{"type":"normal","mu":0,"sigma":1}]


    def to_cpp_params(self) -> dict:
        return {
            "type":           "CIR",
            "r_spot":         self.mkt.r_spot,
            "vol":            self.vol,
            "kappa":          self.kappa,
            "theta":          self.theta,
        }
    
    def set_parameters(self,params:dict) -> None:
        pass