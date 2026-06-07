from abc import ABC, abstractmethod
import numpy as np
from mc_engine.paths.base import PathData


class Product(ABC):
    """Abstract base class for all derivative products.

    A product defines which underlyings drive its payoff, its maturity, and
    how to compute the discounted payoff given a set of simulated paths.
    Concrete subclasses implement the ``payoff`` method for the specific
    product structure (European, American, Asian, swaption, etc.).
    """

    @property
    @abstractmethod
    def underlyings(self) -> list[str]:
        """List of asset identifiers whose paths are required for the payoff.

        These keys must match the keys in the ``processes`` dict passed to the
        engine and in the ``paths`` dict passed to ``payoff``.
        """
        ...

    @property
    @abstractmethod
    def maturity(self) -> float:
        """Time to maturity in years.  Determines the simulation horizon."""
        ...

    @abstractmethod
    def payoff(self, paths: dict[str, PathData],
               discount: float) -> np.ndarray:
        """Compute the discounted payoff for every simulation path.

        Parameters
        ----------
        paths:
            Mapping from asset identifier to its simulated ``PathData``.
        discount:
            Deterministic discount factor D(0, T) = P(0, T) from the
            risk-free curve (used by equity products).  Interest-rate products
            typically use the stochastic discount factor from the paths instead.

        Returns
        -------
        np.ndarray
            1-D array of shape ``[n_sims]`` containing the discounted payoff
            for each simulation path.
        """
        ...
