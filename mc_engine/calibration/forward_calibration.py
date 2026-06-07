import pandas as pd
from sklearn.decomposition import PCA
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
import numpy as np


@dataclass
class HJMVolComponents:
    """PCA volatility structure for the HJM forward-rate model.

    Produced by ``HJMCalibrator`` and consumed by ``HJMProcess``.

    Attributes
    ----------
    tenors:
        Tenor grid knots (in years) used during calibration.
    parameters:
        Cubic spline coefficients for each PCA component — shape
        ``[n_components, n_intervals, 4]``.
    scores_std:
        Standard deviations of the PCA score series — used to scale each
        volatility component to its historical magnitude.
    n_components:
        Number of PCA components retained to reach the target explained
        variance.
    """

    tenors:       np.ndarray
    parameters:   np.ndarray
    scores_std:   np.ndarray
    n_components: int


class HJMCalbirator:
    """Calibrate the HJM volatility structure via PCA on forward-rate changes.

    The calibrator:

    1. Computes daily (or periodic) forward-rate changes from a panel of
       historical forward rates.
    2. Applies PCA and retains the minimum number of components needed to
       explain ``explained_var`` fraction of the total variance.
    3. Fits a natural cubic spline to each principal component over the tenor
       grid so the C++ engine can evaluate the volatility at arbitrary maturities.

    Parameters
    ----------
    forward_data:
        DataFrame of forward rates.  Columns are tenor labels (numeric),
        rows are time-series observations.
    explained_var:
        Minimum fraction of variance that the retained PCA components must
        explain (default 0.90 = 90 %).
    """

    def __init__(self, forward_data: pd.DataFrame,
                 explained_var: float = 0.9) -> None:
        self.fwd_returns = self.compute_forward_returns(forward_data)
        self.tenors      = np.array(self.fwd_returns.columns)
        self.perform_pca_on_forward_returns(explained_var)

    def create_vol_components(self) -> HJMVolComponents:
        """Return the calibrated volatility components as an HJMVolComponents object."""
        return HJMVolComponents(
            tenors       = self.tenors,
            parameters   = self.vol_component_splines,
            scores_std   = self.scores_std,
            n_components = self.n_components,
        )

    def compute_forward_returns(self, fwd_df: pd.DataFrame) -> pd.DataFrame:
        """Compute first differences of forward rates and drop the initial NaN row."""
        return fwd_df.diff().dropna()

    def perform_pca_on_forward_returns(self, explained_var: float = 0.9) -> None:
        """Run PCA and fit splines to the retained principal components.

        Iteratively adds one component at a time until the cumulative explained
        variance threshold is reached.  Each component is then represented as a
        natural cubic spline over the tenor grid.

        Parameters
        ----------
        explained_var:
            Target fraction of explained variance (e.g. 0.9 for 90 %).
        """
        explained_variance_ratio = 0.0
        n_comp = 0
        while explained_variance_ratio < explained_var:
            n_comp += 1
            pca = PCA(n_components=n_comp)
            principal_components = pca.fit_transform(self.fwd_returns)
            explained_variance_ratio = pca.explained_variance_ratio_.sum()

        # Fit a natural cubic spline per component for C++ evaluation
        spline_params = []
        for component in pca.components_:
            spline = CubicSpline(self.tenors, component, bc_type="natural")
            spline_params.append(spline.c.T)   # store polynomial coefficients

        self.vol_component_splines = np.array(spline_params)

        scores = pd.DataFrame(
            data    = principal_components,
            index   = self.fwd_returns.index,
            columns = [f"PC{i+1}_score" for i in range(n_comp)],
        )
        self.n_components = n_comp
        # Standard deviation of each score series scales the vol components
        self.scores_std = scores.std(axis=0)
