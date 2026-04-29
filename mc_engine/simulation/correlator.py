import numpy as np

class Correlator:

    def correlate(self, z: np.ndarray,
                  chol: np.ndarray) -> np.ndarray:
        """
        z:    [n_sims, n_steps, n_assets] — unkorreliert N(0,1)
        chol: [n_assets, n_assets]        — untere Cholesky-Matrix
        →     [n_sims, n_steps, n_assets] — korreliert
        """
        return z @ chol.T