from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize


class Calibrator(ABC):
    """Abstract base class for model calibrators.

    Concrete calibrators minimise the RMSE between model-implied and market
    implied volatilities (or log-likelihoods for time-series calibrators) using
    L-BFGS-B.  Subclasses implement the four abstract methods to define the
    objective function, initial parameter guess, parameter bounds, and result
    parsing.
    """

    def calibrate(self) -> dict:
        """Run the optimisation and return calibrated parameters.

        Uses L-BFGS-B with tight function-value tolerance.  Prints a warning
        if the optimiser does not converge, but still returns the best
        parameters found.

        Returns
        -------
        dict
            Calibrated parameters keyed by name.
        """
        result = minimize(
            self._objective,
            self._initial_params(),
            bounds  = self._bounds(),
            method  = "L-BFGS-B",
            options = {"ftol": 1e-8, "maxiter": 1000},
        )
        if not result.success:
            print(f"Warning: calibration did not converge — {result.message}")
        return self._parse_result(result.x)

    @abstractmethod
    def _objective(self, params: np.ndarray) -> float:
        """Scalar loss function to minimise (lower is better)."""
        ...

    @abstractmethod
    def _initial_params(self) -> list:
        """Starting parameter vector for the optimiser."""
        ...

    @abstractmethod
    def _bounds(self) -> list:
        """List of (lower, upper) bounds for each parameter."""
        ...

    @abstractmethod
    def _parse_result(self, x: np.ndarray) -> dict:
        """Convert the raw optimiser output vector to a named parameter dict."""
        ...

    def _rmse(self, model_vols: np.ndarray,
              market_vols: np.ndarray) -> float:
        """Root-mean-square error between model and market implied vols.

        NaN pairs are silently dropped before computing the statistic.

        Parameters
        ----------
        model_vols:
            Model-implied volatilities.
        market_vols:
            Observed market implied volatilities.

        Returns
        -------
        float
        """
        diff = np.array(model_vols) - np.array(market_vols)
        diff = diff[~np.isnan(diff)]
        return np.sqrt(np.mean(diff**2))
