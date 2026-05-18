import numpy as np
from mc_engine.processes.base import StochasticProcess
from mc_engine.market.curves import YieldCurve


class G2ppProcess(StochasticProcess):
    """
    Zeitunabhängiges G2++ Modell (zwei-Faktor Hull-White).

    """

    process_type = "G2PP"
    noise_dim    = 2

    def __init__(self, curve: YieldCurve,
                 a: float, b: float,
                 sigma: float, eta: float,
                 rho: float,
                 x0: float = 0.0, y0: float = 0.0):
        self.curve = curve
        self.a     = a
        self.b     = b
        self.sigma = sigma
        self.eta   = eta
        self.rho   = rho
        self.x0    = x0
        self.y0    = y0
        self.noise = [{"type":"normal","mu":0,"sigma":1},
                      {"type":"normal","mu":0,"sigma":1}]

    def to_cpp_params(self) -> dict:
        return {
            "type":  "G2PP",
            "a":     self.a,
            "b":     self.b,
            "sigma": self.sigma,
            "eta":   self.eta,
            "rho":   self.rho,
            "x0":    self.x0,
            "y0":    self.y0,
            "curve_tenors": self.curve.tenors.tolist(),
            "curve_rates":  self.curve.rates.tolist(),
            "path_type":      "spot"
        }
    def set_parameters(self,params:dict) -> None:
        pass