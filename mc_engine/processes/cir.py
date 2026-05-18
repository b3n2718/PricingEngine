from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import FIMarketData

class CIRProcess(StochasticProcess):
    """
    Cox-Ingersoll-Ross Process for short rate modelling
    """
    process_type = "CIR"
    noise_dim    = 1

    def __init__(self, mkt: FIMarketData, vol: float, kappa: float, theta: float):
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
            "path_type":      "spot"
        }
    
    def set_parameters(self,params:dict) -> None:
        pass