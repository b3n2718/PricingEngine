import numpy as np
from mc_engine.paths.base import PathData


class ScalarPath(PathData):
    """Concrete PathData implementation for single-factor models (e.g. GBM).

    Wraps a 2-D array of simulated scalar values with shape
    ``[n_sims, n_steps]``, where each row is one simulation path and each
    column is a discrete time step.
    """

    def __init__(self, data: np.ndarray):
        """
        Parameters
        ----------
        data:
            Simulated price/rate paths with shape ``[n_sims, n_steps]``.
        """
        self._data = data

    @property
    def n_sims(self) -> int:
        """Number of Monte Carlo simulation paths."""
        return self._data.shape[0]

    def terminal_value(self) -> np.ndarray:
        """Return the value at the final time step for every simulation.

        Returns
        -------
        np.ndarray
            1-D array of shape ``[n_sims]`` containing each path's last value.
        """
        return self._data[:, -1]

    def at_step(self, step: int) -> np.ndarray:
        """Return the value at a specific time-step index for every simulation.

        Parameters
        ----------
        step:
            Zero-based column index into the path array.

        Returns
        -------
        np.ndarray
            1-D array of shape ``[n_sims]``.
        """
        return self._data[:, step]

    def full_price_path(self) -> np.ndarray:
        """Return the complete path array for all simulations and time steps.

        Returns
        -------
        np.ndarray
            2-D array of shape ``[n_sims, n_steps]``.
        """
        return self._data[:, :]
    
