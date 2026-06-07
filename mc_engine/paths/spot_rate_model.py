import numpy as np
from mc_engine.paths.base import PathData


class SpotRateModel(PathData):
    """Path container for scalar short-rate models (Vasicek, CIR).

    Wraps a 2-D array of simulated short rates with shape
    ``[n_sims, n_steps]`` and provides discount factors and zero-coupon bond
    prices computed analytically from the model parameters — no nested
    simulation is required.

    Parameters
    ----------
    data:
        Simulated short-rate paths with shape ``[n_sims, n_steps]``.
    params:
        Parameter dict from ``StochasticProcess.to_cpp_params()`` augmented
        with ``"dt"`` (time step size).
    """

    def __init__(self, data: np.ndarray, params: dict):
        self._data  = data
        self.params = params

    @property
    def n_sims(self) -> int:
        """Number of Monte Carlo simulation paths."""
        return self._data.shape[0]

    @property
    def n_steps(self) -> int:
        """Number of discrete time steps per path."""
        return self._data.shape[1]

    def terminal_value(self) -> np.ndarray:
        """Short rate at the final time step — shape ``[n_sims]``."""
        return self._data[:, -1]

    def at_step(self, step: int) -> np.ndarray:
        """Short rate at a given time-step index — shape ``[n_sims]``."""
        return self._data[:, step]

    def full_path(self) -> np.ndarray:
        """Return all simulated paths — shape ``[n_sims, n_steps]``."""
        return self._data[:, :]

    def df(self) -> np.ndarray:
        """Discount factor from time 0 to the simulation horizon.

        Approximates exp(-∫₀ᵀ r(t) dt) via the Euler sum of short rates.

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        return np.exp(-self._data.sum(axis=1) * self.params["dt"])

    def zcp(self, t1: float, t2: float) -> np.ndarray:
        """Zero-coupon bond price P(t1, t2) using the closed-form formula.

        Both Vasicek and CIR admit affine term-structure solutions of the form

            P(t1, t2) = A(t1, t2) · exp(-B(t1, t2) · r(t1))

        The coefficients are computed from the calibrated model parameters.

        Parameters
        ----------
        t1:
            Valuation time (years).
        t2:
            Maturity of the bond (years), must satisfy t2 > t1.

        Returns
        -------
        np.ndarray
            Bond prices across all paths — shape ``[n_sims]``.
        """
        index_t1 = int(t1 / self.params["dt"] - 1)

        if self.params["type"] == "VASICEK":
            B_t_T = (1 - np.exp(-self.params["kappa"] * (t2 - t1))) / self.params["kappa"]
            ln_A  = (
                (self.params["theta"] - self.params["vol"]**2 / (2 * self.params["kappa"]**2))
                * (B_t_T - (t2 - t1))
                - self.params["vol"]**2 / (4 * self.params["kappa"]) * B_t_T**2
            )
            return np.exp(ln_A - B_t_T * self._data[:, index_t1])

        if self.params["type"] == "CIR":
            gamma = np.sqrt(self.params["kappa"]**2 + 2 * self.params["vol"]**2)
            B_t_T = (
                2 * (np.exp(gamma * (t2 - t1)) - 1)
                / ((gamma + self.params["kappa"]) * (np.exp(gamma * (t2 - t1)) - 1) + 2 * gamma)
            )
            ln_A  = (
                2 * self.params["kappa"] * self.params["theta"] / self.params["vol"]**2
                * np.log(
                    2 * gamma * np.exp((gamma + self.params["kappa"]) * (t2 - t1) / 2)
                    / ((gamma + self.params["kappa"]) * (np.exp(gamma * (t2 - t1)) - 1) + 2 * gamma)
                )
            )
            return np.exp(ln_A - B_t_T * self._data[:, index_t1])
