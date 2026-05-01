import numpy as np
from scipy.stats import qmc
from enum import Enum
from scipy.stats import gamma, norm

class RNGType(Enum):
    PSEUDO = "pseudo"
    SOBOL  = "sobol"

class RandomNumberGenerator:

    def __init__(self, rng_type: RNGType = RNGType.SOBOL, seed: int = 42):
        self.rng_type = rng_type
        self.seed     = seed
            
    def  _generate_sobol(self, n_sims: int, n_steps: int, total_noise_dim: int):
        d       = n_steps * total_noise_dim
        sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
        uniform = sampler.random(n_sims)
        samples = np.clip(uniform, 1e-10, 1 - 1e-10)
        return samples.reshape(n_sims, n_steps, total_noise_dim)
    
    def _generate_pseudo(self, n_sims, n_steps, total_noise_dim) -> np.ndarray:
        samples = np.random.uniform(size=n_sims*n_steps*total_noise_dim).reshape(n_sims, n_steps, total_noise_dim)
        return np.clip(samples, 1e-10, 1 - 1e-10)
    
    def generate(self, n_sims: int, n_steps: int,
                 noise_dim: list[dict]) -> np.ndarray:
        match self.rng_type:
            case RNGType.PSEUDO:
                z = self._generate_pseudo(n_sims, n_steps, len(noise_dim))
            case RNGType.SOBOL:
                z = self._generate_sobol(n_sims, n_steps, len(noise_dim))

        for i,dim in enumerate(noise_dim):
            match dim["type"]:
                case "uniform":
                    pass
                case "normal":
                    z[:,:,i] = norm.ppf(z[:,:,i],dim["mu"],dim["sigma"])
                case "gamma":
                    z[:,:,i] = gamma.ppf(z[:,:,i],dim["a"],dim["b"])
        return z