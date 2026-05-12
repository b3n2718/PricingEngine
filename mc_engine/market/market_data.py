from dataclasses import dataclass
from market.curves import YieldCurve

@dataclass
class EquityMarketData:
    spot:      float
    div_yield: float
    curve:     YieldCurve

@dataclass
class FISpotRateMarketData:
    r_spot:      float

@dataclass
class FIForwardRateMarketData:
    r_forward:      float