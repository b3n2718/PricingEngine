import numpy as np
from mc_engine.paths.base import PathData


class ForwardRateModel(PathData):
    """Path container for the HJM forward-rate model.

    Wraps a 3-D array of simulated forward curves with shape
    ``[n_sims, n_tenors, n_steps]``.  The first tenor slice (index 0) contains
    the instantaneous short rate, which is used for discounting.

    Parameters
    ----------
    data:
        Simulated forward-rate paths with shape ``[n_sims, n_tenors, n_steps]``.
    params:
        Parameter dict from ``HJMProcess.to_cpp_params()`` augmented with
        ``"dt"`` and ``"tenors"``.
    """

    def __init__(self, data: np.ndarray, params: dict):
        self._data   = data
        self.params  = params
        self._dt     = params["dt"]
        self._tenors = np.array(params["tenors"]).astype(float)

    @property
    def n_sims(self) -> int:
        """Number of Monte Carlo simulation paths."""
        return self._data.shape[0]

    @property
    def n_steps(self) -> int:
        """Number of discrete time steps."""
        return self._data.shape[2]

    @property
    def n_tenors(self) -> int:
        """Number of tenor grid points in the forward curve."""
        return self._data.shape[1]

    # ── PathData interface ─────────────────────────────────────────────────

    def terminal_value(self) -> np.ndarray:
        """Short rate at the final time step — shape ``[n_sims]``."""
        return self._data[:, -1, 0]

    def at_step(self, step: int) -> np.ndarray:
        """Short rate at a given time-step index — shape ``[n_sims]``."""
        return self._data[:, step, 0]

    # ── HJM-specific helpers ───────────────────────────────────────────────

    def forward_curve_at(self, step: int) -> np.ndarray:
        """Full forward curve at a given time step.

        Returns
        -------
        np.ndarray
            Shape ``[n_sims, n_tenors]``.
        """
        return self._data[:, step, :]

    def full_path(self) -> np.ndarray:
        """Return the complete raw path array — shape ``[n_sims, n_tenors, n_steps]``."""
        return self._data

    def df(self) -> np.ndarray:
        """Discount factor from time 0 to the simulation horizon.

        Integrates the short rate (first tenor) numerically via the trapezoid
        rule on a uniform time grid:

            D(0, T) = exp(-∫₀ᵀ r(t) dt)

        Returns
        -------
        np.ndarray
            Shape ``[n_sims]``.
        """
        short_rates = self._data[:, 0, :]                      # [n_sims, n_steps]
        times       = np.arange(self.n_steps) * self._dt
        integral    = np.trapezoid(short_rates, times, axis=1)
        return np.exp(-integral)

    def zcp(self, t1: float, t2: float) -> np.ndarray:
        """Zero-coupon bond price P(t1, t2) from the simulated forward curve.

        Uses the HJM relation:

            P(t1, t2) = exp(-∫_{t1}^{t2} f(t1, s) ds)

        The integral is evaluated numerically with trapezoid quadrature on the
        non-uniform tenor grid.  Linear interpolation is applied when t2 - t1
        does not fall exactly on a grid knot.

        Parameters
        ----------
        t1:
            Valuation time (years).
        t2:
            Bond maturity (years), must satisfy t2 > t1.

        Returns
        -------
        np.ndarray
            Bond prices across all paths — shape ``[n_sims]``.

        Raises
        ------
        ValueError
            If t1 is beyond the simulation horizon or t2 - t1 exceeds the
            maximum tenor in the grid.
        """
        tau  = t2 - t1
        step = min(int(round(t1 / self._dt)), self._data.shape[1] - 1)

        if step >= self.n_steps:
            raise ValueError(
                f"t1={t1} is beyond the simulation horizon "
                f"(max={self.n_steps * self._dt:.2f})"
            )

        if tau > self._tenors[-1]:
            raise ValueError(
                f"t2-t1={tau:.2f} exceeds the tenor grid "
                f"(max={self._tenors[-1]:.2f}). "
                f"Extend the HJM tenor grid accordingly."
            )

        f_t1     = self._data[:, :, step]      # [n_sims, n_tenors]

        # Restrict to tenor points up to tau
        mask     = self._tenors <= tau
        tenors_r = self._tenors[mask]
        f_slice  = f_t1[:, mask]

        # Linearly interpolate the forward rate at the exact boundary tau
        if tenors_r[-1] < tau:
            idx     = mask.sum() - 1
            t_left  = self._tenors[idx]
            t_right = self._tenors[idx + 1]
            f_left  = f_t1[:, idx]
            f_right = f_t1[:, idx + 1]
            w        = (tau - t_left) / (t_right - t_left)
            f_interp = f_left + w * (f_right - f_left)

            tenors_r = np.append(tenors_r, tau)
            f_slice  = np.concatenate([f_slice, f_interp[:, np.newaxis]], axis=1)

        integral = np.trapezoid(f_slice, tenors_r, axis=1)
        return np.exp(-integral)
