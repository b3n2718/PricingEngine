import numpy as np
from paths.base import PathData
from paths.scalar_path import ScalarPath
from mc_engine.paths.spot_rate_model import SpotRateModel

def build_path(raw: np.ndarray, process_type: str, params:dict) -> PathData:
    match process_type:
        case "GBM":
            return ScalarPath(raw)
        case "HESTON":
            return ScalarPath(raw)   # Heston gibt ebenfalls nur Spot zurück
        case "GAMMAVARIANCE":
            return ScalarPath(raw)
        case "VASICEK":
            return SpotRateModel(raw,params)
        case "CIR":
            return SpotRateModel(raw,params)
        case _:
            raise ValueError(f"Unbekannter Prozesstyp: {process_type}")