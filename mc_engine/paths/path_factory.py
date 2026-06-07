import numpy as np
from mc_engine.paths.base import PathData
from mc_engine.paths.scalar_path import ScalarPath
from mc_engine.paths.spot_rate_model import SpotRateModel
from mc_engine.paths.forward_rate_model import ForwardRateModel
from mc_engine.paths.gaussian_two_factors_path import G2ppPath


def build_path(raw: np.ndarray, process_type: str, params: dict) -> PathData:
    """Factory: wrap a raw simulation array in the appropriate PathData subclass.

    The engine calls this function after the C++ path generator returns raw
    NumPy arrays.  Each process type maps to a dedicated container class that
    exposes the correct pricing interface (e.g. ``zcp``, ``df``).

    Parameters
    ----------
    raw:
        Raw simulated data from the C++ backend.  Shape and interpretation
        depend on the process type.
    process_type:
        String key matching ``StochasticProcess.process_type``.
    params:
        Parameter dict (from ``to_cpp_params()``) augmented with ``"dt"``.

    Returns
    -------
    PathData
        Typed path container for the given process.

    Raises
    ------
    ValueError
        If the process type is not recognised.
    """
    match process_type:
        case "GBM":
            return ScalarPath(raw)
        case "HESTON":
            # Heston returns only the spot path; the variance path is internal
            return ScalarPath(raw)
        case "VARIANCEGAMMA":
            return ScalarPath(raw)
        case "VASICEK":
            return SpotRateModel(raw, params)
        case "CIR":
            return SpotRateModel(raw, params)
        case "HJM":
            return ForwardRateModel(raw, params)
        case "G2PP":
            return G2ppPath(raw, params)
        case _:
            raise ValueError(f"Unknown process type: {process_type}")
