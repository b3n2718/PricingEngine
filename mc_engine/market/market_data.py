from dataclasses import dataclass
from mc_engine.market.curves import YieldCurve
import numpy as np

@dataclass
class EquityMarketData:
    spot:      float
    div_yield: float
    curve:     YieldCurve

@dataclass
class FIMarketData:
    r_spot:      float
    r_forward:      np.array
    forward_tenors: np.array 
       