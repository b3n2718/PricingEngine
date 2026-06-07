from abc import ABC, abstractmethod
import numpy as np


class PathData(ABC):
    """Abstract base class for Monte Carlo simulation path containers.

    Subclasses wrap the raw simulated data (scalar, multi-factor, etc.) and
    expose a uniform interface so that pricers can query path values without
    knowing the underlying storage layout.
    """

    @property
    @abstractmethod
    def n_sims(self) -> int:
        """Number of Monte Carlo simulation paths."""
        ...

    @abstractmethod
    def terminal_value(self) -> np.ndarray:
        """Return the terminal (maturity) value for every simulation path.

        Returns
        -------
        np.ndarray
            1-D array of shape ``[n_sims]``.
        """
        ...

    @abstractmethod
    def at_step(self, step: int) -> np.ndarray:
        """Return the value at a given discrete time-step index for every path.

        Parameters
        ----------
        step:
            Zero-based time-step index.

        Returns
        -------
        np.ndarray
            1-D array of shape ``[n_sims]``.
        """
        ...