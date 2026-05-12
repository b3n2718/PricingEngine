from processes.base import StochasticProcess
from market.market_data import FIForwardRateMarketData
from calibration.HJM.forward_calibration import HJMVolComponents
class HJMProcess(StochasticProcess):
    process_type = "HJM"
    

    def __init__(self, mkt: FIForwardRateMarketData, vol_component: HJMVolComponents):
        self.mkt = mkt
        self.vol = vol_component
        self.noise = [{"type":"normal","mu":0,"sigma":1}*len(vol_component)]
        self.noise_dim  = len(vol_component)
        self.vol_component = vol_component


    def to_cpp_params(self) -> dict:
        return {
            "type":           "HJM",
            "r_forward":      self.mkt.r_forward,
            "std_scores":     self.vol_component.parameters,
            "std_scores":     self.vol_component.scores_std,
            "tenors":         self.vol_component.tenors,
            "num_vol_comp":   self.noise_dim
        }
    
    def set_parameters(self,params:dict) -> None:
        pass