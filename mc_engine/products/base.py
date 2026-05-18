from abc import ABC, abstractmethod
import numpy as np
from mc_engine.paths.base import PathData

class Product(ABC):

    @property
    @abstractmethod
    def underlyings(self) -> list[str]: ...

    @property
    @abstractmethod
    def maturity(self) -> float: ...

    @abstractmethod
    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray: ...