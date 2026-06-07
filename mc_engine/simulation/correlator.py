import numpy as np


class Correlator:
    """Applies a Cholesky factor to introduce cross-asset correlations.

    Given independent standard-normal draws ``z``, the correlated draws are
    computed as ``z @ chol.T``, where ``chol`` is the lower-triangular Cholesky
    decomposition of the desired correlation matrix.
    """

    def correlate(self, z: np.ndarray, chol: np.ndarray) -> np.ndarray:
        """Transform uncorrelated draws into correlated draws.

        Parameters
        ----------
        z:
            Uncorrelated standard-normal draws with shape
            ``[n_sims, n_steps, n_noise_dims]``.
        chol:
            Lower-triangular Cholesky factor of the correlation matrix,
            shape ``[n_noise_dims, n_noise_dims]``.

        Returns
        -------
        np.ndarray
            Correlated draws with the same shape as ``z``.
        """
        return z @ chol.T
