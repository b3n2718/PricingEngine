import numpy as np
from paths.base import PathData

class ScalarPath(PathData):
    """Für GBM — raw array shape [n_sims, n_steps]"""

    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    def n_sims(self) -> int:
        return self._data.shape[0]

    def terminal_value(self) -> np.ndarray:
        return self._data[:, -1]

    def at_step(self, step: int) -> np.ndarray:
        return self._data[:, step]