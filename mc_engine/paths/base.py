from abc import ABC, abstractmethod
import numpy as np

class PathData(ABC):

    @property
    @abstractmethod
    def n_sims(self) -> int: ...

    @abstractmethod
    def terminal_value(self) -> np.ndarray:
        """Endwert für alle Simulationen — shape [n_sims]"""
        ...

    @abstractmethod
    def at_step(self, step: int) -> np.ndarray:
        """Wert zum Zeitschritt-Index — shape [n_sims]"""
        ...