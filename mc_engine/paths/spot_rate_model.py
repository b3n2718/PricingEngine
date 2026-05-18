import numpy as np
from mc_engine.paths.base import PathData


class SpotRateModel(PathData):
    """Für GBM — raw array shape [n_sims, n_steps]"""

    def __init__(self, data: np.ndarray,params: dict):
        self._data = data
        self.params = params

    @property
    def n_sims(self) -> int:
        return self._data.shape[0]

    @property
    def n_steps(self) -> int:
        return self._data.shape[1]
    
    def terminal_value(self) -> np.ndarray:
        return self._data[:, -1]

    def at_step(self, step: int) -> np.ndarray:
        return self._data[:, step]
    
    def full_path(self) -> np.ndarray:
        return self._data[:, :]
    
    def df(self):
        return np.exp(-self._data.sum(axis=1)*self.params["dt"])
    
    def zcp(self,t1:float,t2:float):
        index_t1 = int(t1/self.params["dt"] -1)
        index_t2 = int(t2/self.params["dt"]) 
        if self.params["type"] == "VASICEK":        
            B_t_T = (1- np.exp(-self.params["kappa"]*(t2-t1)))/self.params["kappa"]
            ln_A = (self.params["theta"] - self.params["vol"]**2/(2*self.params["kappa"]**2)) * (B_t_T - (t2 - t1)) -self.params["vol"]**2/(4*self.params["kappa"]) * B_t_T**2
            return np.exp(ln_A-B_t_T * self._data[:, index_t1])
        if self.params["type"] == "CIR":
            gamma = np.sqrt(self.params["kappa"]**2+2*self.params["vol"]**2)
            B_t_T = (2*np.exp(gamma*(t2-t1))-1)/((gamma+self.params["kappa"])*(np.exp(gamma*(t2-t1))-1)+2*gamma)
            ln_A = 2*self.params["kappa"]*self.params["theta"]/(self.params["vol"]**2) *np.log(
                (2*gamma*np.exp((gamma+self.params["kappa"])*(t2-t1)/2)/((gamma+self.params["kappa"])*(np.exp(gamma*(t2-t1))-1)+2*gamma))
            )
            return np.exp(ln_A-B_t_T * self._data[:, index_t1])
            