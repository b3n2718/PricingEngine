from dataclasses import dataclass
from mc_engine.market.curves import YieldCurve
import numpy as np


@dataclass
class EquityMarketData:
    """Market data container for equity models (GBM, Heston, Variance-Gamma).

    Attributes
    ----------
    spot:
        Current spot price of the underlying asset.
    div_yield:
        Continuous dividend yield (annualised).
    curve:
        Risk-free yield curve used for discounting and as the drift input.
    """

    spot:      float
    div_yield: float
    curve:     YieldCurve


@dataclass
class FIMarketData:
    """Market data container for fixed-income models (Vasicek, CIR, HJM, G2++).

    Attributes
    ----------
    r_spot:
        Current instantaneous short rate.
    r_forward:
        Array of initial forward rates on the tenor grid.
    forward_tenors:
        Array of tenors (in years) corresponding to ``r_forward``.
    """

    r_spot:         float
    r_forward:      np.ndarray
    forward_tenors: np.ndarray
