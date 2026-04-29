from dataclasses import dataclass, field
from simulation.rng import RNGType
import numpy as np

@dataclass
class MCConfig:
    n_sims:          int     = 100_000
    n_steps:         int     = 252
    rng_type:        RNGType = RNGType.PSEUDO
    seed:            int     = 42
    use_antithetic:  bool    = True
    corrolation_matrix: np.arry = None