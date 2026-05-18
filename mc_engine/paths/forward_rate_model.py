import numpy as np
from mc_engine.paths.base import PathData


class ForwardRateModel(PathData):
    """
    HJM Forward-Rate Modell mit nicht-uniformem Tenor-Grid.
    data shape: [n_sims, n_steps, n_tenors]
    """

    def __init__(self, data: np.ndarray, params: dict):
        self._data   = data
        self.params  = params
        self._dt     = params["dt"]
        self._tenors = np.array(params["tenors"]).astype(float)   # nicht-uniformes Tenor-Grid

    @property
    def n_sims(self) -> int:
        return self._data.shape[0]

    @property
    def n_steps(self) -> int:
        return self._data.shape[2]

    @property
    def n_tenors(self) -> int:
        return self._data.shape[1]

    # ── PathData Interface ──────────────────────────────────────────

    def terminal_value(self) -> np.ndarray:
        return self._data[:, -1, 0]

    def at_step(self, step: int) -> np.ndarray:
        return self._data[:, step, 0]

    # ── HJM-spezifisch ─────────────────────────────────────────────

    def forward_curve_at(self, step: int) -> np.ndarray:
        """Volle Forward-Kurve zum Zeitschritt — [n_sims, n_tenors]."""
        return self._data[:, step, :]

    def full_path(self) -> np.ndarray:
        return self._data

    def df(self) -> np.ndarray:
        """
        Diskontfaktor von 0 bis T_end.
        ∫_0^T r(t) dt — numerisch über nicht-uniformes Zeitgitter.
        shape: [n_sims]
        """
        short_rates = self._data[:, 0, :]             # [n_sims, n_steps]
        times       = np.arange(self.n_steps) * self._dt   # uniformes Zeitgitter

        integral    = np.trapezoid(short_rates, times, axis=1)
        return np.exp(-integral)                      # [n_sims]

    def zcp(self, t1: float, t2: float) -> np.ndarray:
        """
        P(t1, t2) = exp(-∫_{t1}^{t2} f(t1, s) ds)

        Nicht-uniformes Tenor-Grid — np.trapezoid mit echten Tenor-Werten.
        shape: [n_sims]
        """
        tau    = t2 - t1
        step   = min(int(round(t1 / self._dt)), self._data.shape[1] - 1)
        

        if step >= self.n_steps:
            raise ValueError(
                f"t1={t1} liegt außerhalb des Simulationshorizonts "
                f"(max={self.n_steps * self._dt:.2f})"
            )

        if tau > self._tenors[-1]:
            raise ValueError(
                f"t2-t1={tau:.2f} liegt außerhalb des Tenor-Grids "
                f"(max={self._tenors[-1]:.2f}). "
                f"Tenor-Grid beim HJM-Setup entsprechend wählen."
            )

        f_t1 = self._data[:, :, step]        # [n_sims, n_tenors]

        # Relevante Tenor-Punkte bis tau
        mask     = self._tenors <= tau
        tenors_r = self._tenors[mask]
        f_slice  = f_t1[:, mask]             # [n_sims, n_relevant_tenors]

        # Letzten Punkt interpolieren falls tau nicht genau auf Grid liegt
        if tenors_r[-1] < tau:
            idx     = mask.sum() - 1
            t_left  = self._tenors[idx]
            t_right = self._tenors[idx + 1]
            f_left  = f_t1[:, idx]
            f_right = f_t1[:, idx + 1]
            # lineare Interpolation
            w        = (tau - t_left) / (t_right - t_left)
            f_interp = f_left + w * (f_right - f_left)

            tenors_r = np.append(tenors_r, tau)
            f_slice  = np.concatenate(
                [f_slice, f_interp[:, np.newaxis]], axis=1
            )

        integral = np.trapezoid(f_slice, tenors_r, axis=1)
        return np.exp(-integral)             # [n_sims]