from dataclasses import dataclass, field
from mc_engine.simulation.rng import RNGType
import numpy as np


@dataclass
class MCConfig:
    """Configuration for a Monte Carlo pricing run.

    Attributes
    ----------
    n_sims:
        Total number of simulation paths (default 100 000).
    n_steps:
        Number of discrete time steps per path (default 252, i.e. daily).
    rng_type:
        Random-number generator backend — ``RNGType.PSEUDO`` (standard
        pseudo-random) or ``RNGType.SOBOL`` (quasi-random Sobol sequence for
        lower variance).
    seed:
        Random seed for reproducibility (default 42).
    use_antithetic:
        If True, the engine generates ``n_sims // 2`` paths and mirrors them
        (antithetic variates variance-reduction technique).
    corrolation_matrix:
        Lower-triangular Cholesky matrix for cross-asset correlation.  Set to
        None for a single underlying or when assets are assumed independent.
    """

    n_sims:             int       = 100_000
    n_steps:            int       = 252
    rng_type:           RNGType   = RNGType.PSEUDO
    seed:               int       = 42
    use_antithetic:     bool      = False
    corrolation_matrix: np.ndarray = None
