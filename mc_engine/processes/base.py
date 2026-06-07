from abc import ABC, abstractmethod


class StochasticProcess(ABC):
    """Abstract base class for all stochastic processes.

    Every concrete process (GBM, Heston, CIR, etc.) must subclass this and
    implement the three abstract members below.  The engine uses these to
    serialize parameters for the C++ path-generator and to configure the
    random-number dimensions required by each model.
    """

    @property
    @abstractmethod
    def process_type(self) -> str:
        """Unique string key identifying the process (e.g. ``"GBM"``).

        The key must match the corresponding C++ implementation registered in
        the path generator.
        """
        ...

    @abstractmethod
    def to_cpp_params(self) -> dict:
        """Serialize model parameters into a dict for the C++ path generator.

        Returns
        -------
        dict
            Flat dictionary of parameter names to values.  Required keys vary
            by process type but always include ``"type"`` and ``"path_type"``.
        """
        ...

    @abstractmethod
    def set_parameters(self, params: dict) -> None:
        """Apply engine-level parameters (e.g. ``dt``) before simulation.

        Called by the engine once per pricing run so that processes whose noise
        dimensions depend on the time-step (e.g. Variance-Gamma gamma draws)
        can update their ``noise`` list accordingly.

        Parameters
        ----------
        params:
            Dictionary containing at least ``"dt"`` (the simulation time step).
        """
        ...
