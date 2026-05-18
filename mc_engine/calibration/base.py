from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize


class Calibrator(ABC):

    def calibrate(self) -> dict:
        result = minimize(
            self._objective,
            self._initial_params(),
            bounds  = self._bounds(),
            method  = "L-BFGS-B",
            options = {"ftol": 1e-8, "maxiter": 1000}
        )
        if not result.success:
            print(f"Warnung: Kalibrierung nicht konvergiert — {result.message}")
        return self._parse_result(result.x)

    @abstractmethod
    def _objective(self, params: np.ndarray) -> float: ...

    @abstractmethod
    def _initial_params(self) -> list: ...

    @abstractmethod
    def _bounds(self) -> list: ...

    @abstractmethod
    def _parse_result(self, x: np.ndarray) -> dict: ...

    def _rmse(self, model_vols: np.ndarray,
              market_vols: np.ndarray) -> float:
        diff = np.array(model_vols) - np.array(market_vols)
        diff = diff[~np.isnan(diff)]
        return np.sqrt(np.mean(diff**2))