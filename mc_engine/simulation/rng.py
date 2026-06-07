import numpy as np
from scipy.stats import qmc
from enum import Enum
from scipy.stats import gamma, norm


class RNGType(Enum):
    """Enumeration of supported random-number generator backends."""

    PSEUDO = "pseudo"  # Standard pseudo-random (NumPy uniform)
    SOBOL  = "sobol"   # Scrambled Sobol quasi-random sequence (lower error)


class RandomNumberGenerator:
    """Generates multi-dimensional random samples for Monte Carlo simulation.

    Supports both pseudo-random (uniform) and quasi-random (Sobol) sampling.
    After generating a uniform base sample, each noise dimension is transformed
    to the required marginal distribution (normal, gamma, etc.) via the
    inverse-CDF (probability integral transform).

    Parameters
    ----------
    rng_type:
        Backend to use: ``RNGType.PSEUDO`` (default) or ``RNGType.SOBOL``.
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, rng_type: RNGType = RNGType.SOBOL, seed: int = 42):
        self.rng_type = rng_type
        self.seed     = seed

    def _generate_sobol(self, n_sims: int, n_steps: int,
                        total_noise_dim: int) -> np.ndarray:
        """Draw a scrambled Sobol sequence reshaped to ``[n_sims, n_steps, dim]``."""
        d       = n_steps * total_noise_dim
        sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
        uniform = sampler.random(n_sims)
        # Clip to avoid degenerate inverse-CDF values at the boundaries
        samples = np.clip(uniform, 1e-10, 1 - 1e-10)
        return samples.reshape(n_sims, n_steps, total_noise_dim)

    def _generate_pseudo(self, n_sims: int, n_steps: int,
                         total_noise_dim: int) -> np.ndarray:
        """Draw pseudo-random uniform samples reshaped to ``[n_sims, n_steps, dim]``."""
        samples = np.random.uniform(
            size=n_sims * n_steps * total_noise_dim
        ).reshape(n_sims, n_steps, total_noise_dim)
        return np.clip(samples, 1e-6, 0.999)

    def generate(self, n_sims: int, n_steps: int,
                 noise_dim: list[dict]) -> np.ndarray:
        """Generate random samples for all noise dimensions.

        Each element of ``noise_dim`` specifies the marginal distribution for
        one noise channel.  The uniform base sample is transformed per-channel
        using the appropriate inverse CDF.

        Parameters
        ----------
        n_sims:
            Number of simulation paths.
        n_steps:
            Number of discrete time steps.
        noise_dim:
            List of dicts, one per noise channel.  Each dict must contain
            ``"type"`` (``"normal"``, ``"gamma"``, or ``"uniform"``) plus the
            relevant distribution parameters (``"mu"``, ``"sigma"`` for normal;
            ``"a"``, ``"b"`` for gamma).

        Returns
        -------
        np.ndarray
            Shape ``[n_sims, n_steps, len(noise_dim)]``.
        """
        match self.rng_type:
            case RNGType.PSEUDO:
                z = self._generate_pseudo(n_sims, n_steps, len(noise_dim))
            case RNGType.SOBOL:
                z = self._generate_sobol(n_sims, n_steps, len(noise_dim))

        # Apply inverse-CDF transform per noise dimension
        for i, dim in enumerate(noise_dim):
            match dim["type"]:
                case "uniform":
                    pass  # already uniform — no transform needed
                case "normal":
                    z[:, :, i] = norm.ppf(z[:, :, i], dim["mu"], dim["sigma"])
                case "gamma":
                    z[:, :, i] = gamma.ppf(z[:, :, i], dim["a"], dim["b"])

        return z
