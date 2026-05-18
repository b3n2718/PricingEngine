from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import FIMarketData
from mc_engine.calibration.forward_calibration import HJMVolComponents
class HJMProcess(StochasticProcess):
    process_type = "HJM"
    

    def __init__(self, mkt: FIMarketData, vol_component: HJMVolComponents):
        self.mkt = mkt
        self.vol = vol_component
        self.noise = [{"type":"normal","mu":0,"sigma":1} for i in range(vol_component.n_components)]
        self.noise_dim  = vol_component.n_components
        self.vol_component = vol_component


    def to_cpp_params(self) -> dict:
        return {
            "type":           "HJM",
            "r_forward":      self.mkt.r_forward.tolist(),
            "spline_parameters":     self.vol_component.parameters,
            "std_scores":     self.vol_component.scores_std,
            "tenors":         self.vol_component.tenors,
            "num_vol_comp":   self.noise_dim,
            "path_type":      "forward"
        }
    
    def set_parameters(self,params:dict) -> None:
        pass