import numpy as np
from scipy.stats import qmc
from enum import Enum

class RNGType(Enum):
    PSEUDO = "pseudo"
    SOBOL  = "sobol"

class RandomNumberGenerator:

    def __init__(self, rng_type: RNGType = RNGType.PSEUDO, seed: int = 42):
        self.rng_type = rng_type
        self.seed     = seed

    def generate(self, n_sims: int, n_steps: int,
                 total_noise_dim: int,rn_type: str="normal") -> np.ndarray:
        """
        Gibt unkorrelierte N(0,1) Zahlen zurück.
        shape: [n_sims, n_steps, total_noise_dim]
        """
        if rn_type == "normal":
            match self.rng_type:
                case RNGType.PSEUDO:
                    return self._pseudo_normal(n_sims, n_steps, total_noise_dim)
                case RNGType.SOBOL:
                    return self._sobol_normal(n_sims, n_steps, total_noise_dim)
        elif rn_type == "uniform":
            match self.rng_type:
                case RNGType.PSEUDO:
                    return self._pseudo_unifrom(n_sims, n_steps, total_noise_dim)
                case RNGType.SOBOL:
                    return self._sobol_unifrom(n_sims, n_steps, total_noise_dim)            

    def _pseudo_normal(self, n_sims, n_steps, total_noise_dim) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.standard_normal((n_sims, n_steps, total_noise_dim))

    def _sobol_normal(self, n_sims, n_steps, total_noise_dim) -> np.ndarray:
        d       = n_steps * total_noise_dim
        sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
        uniform = sampler.random(n_sims)                # [n_sims, d] ∈ (0,1)
        # Inverse CDF → N(0,1)
        samples = np.clip(uniform, 1e-10, 1 - 1e-10)   # Randwerte vermeiden
        samples = qmc.MultivariateNormalQMC._inv_normal(samples)
        return samples.reshape(n_sims, n_steps, total_noise_dim)

    def _pseudo_uniform(self, n_sims, n_steps, total_noise_dim) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.uniform((n_sims, n_steps, total_noise_dim))

    def _sobol_uniform(self, n_sims, n_steps, total_noise_dim) -> np.ndarray:
        d       = n_steps * total_noise_dim
        sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
        uniform = sampler.random(n_sims)                # [n_sims, d] ∈ (0,1)
        # Inverse CDF → N(0,1)
        samples = np.clip(uniform, 1e-10, 1 - 1e-10)   # Randwerte vermeiden
        return samples.reshape(n_sims, n_steps, total_noise_dim)