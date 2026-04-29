from dataclasses import dataclass
from market.curves import YieldCurve

@dataclass
class EquityMarketData:
    spot:      float
    div_yield: float
    curve:     YieldCurve