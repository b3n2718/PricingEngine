import numpy as np
from mc_engine.paths.base import PathData
from mc_engine.paths.scalar_path import ScalarPath
from mc_engine.paths.spot_rate_model import SpotRateModel
from mc_engine.paths.forward_rate_model import ForwardRateModel
from mc_engine.paths.gaussian_two_factors_path import G2ppPath


def build_path(raw: np.ndarray, process_type: str, params:dict) -> PathData:
    match process_type:
        case "GBM":
            return ScalarPath(raw)
        case "HESTON":
            return ScalarPath(raw)   # Heston gibt ebenfalls nur Spot zurück
        case "VARIANCEGAMMA":
            return ScalarPath(raw)
        case "VASICEK":
            return SpotRateModel(raw,params)
        case "CIR":
            return SpotRateModel(raw,params)
        case "HJM":
            return ForwardRateModel(raw,params)
        case "G2PP":
            return G2ppPath(raw,params)
        case _:
            raise ValueError(f"Unbekannter Prozesstyp: {process_type}")