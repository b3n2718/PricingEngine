import numpy as np
from paths.base import PathData
from paths.scalar_path import ScalarPath

def build_path(raw: np.ndarray, process_type: str) -> PathData:
    match process_type:
        case "GBM":
            return ScalarPath(raw)
        case "HESTON":
            return ScalarPath(raw)   # Heston gibt ebenfalls nur Spot zurück
        case _:
            raise ValueError(f"Unbekannter Prozesstyp: {process_type}")