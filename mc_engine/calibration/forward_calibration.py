import pandas as pd
from sklearn.decomposition import PCA
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
import numpy as np

@dataclass
class HJMVolComponents:
    tenors: np.array
    parameters: np.array
    scores_std : np.array
    n_components: float
    

class HJMCalbirator:
    def __init__(self,forward_data:pd.DataFrame, explained_var:float=0.9) -> pd.DataFrame:
        self.fwd_returns = self.compute_forward_returns(forward_data)
        self.tenors = np.array(self.fwd_returns.columns)
        self.perform_pca_on_forward_returns(explained_var)
    
    def create_vol_components(self):
        return HJMVolComponents(tenors=self.tenors, 
                                parameters = self.vol_component_splines,
                                scores_std = self.scores_std,
                                n_components=self.n_components
                                )

    def compute_forward_returns(self,fwd_df: pd.DataFrame) -> pd.DataFrame:
        returns = fwd_df.diff().dropna()
        return returns

    def perform_pca_on_forward_returns(self, explained_var:float=0.9):
        explained_variance_ratio = 0
        n_comp = 0
        while explained_variance_ratio < explained_var:
            n_comp +=1
            pca = PCA(n_components=n_comp)
            principal_components = pca.fit_transform(self.fwd_returns)
            explained_variance_ratio = pca.explained_variance_ratio_.sum()

        spline_params = []

        for component in pca.components_:
            spline = CubicSpline(self.tenors, component, bc_type='natural')
            spline_params.append(spline.c.T)
        self.vol_component_splines = np.array(spline_params)
        scores = pd.DataFrame(
            data=principal_components,
            index=self.fwd_returns.index,
            columns=[f"PC{i+1}_score" for i in range(n_comp)]
        )
        self.n_components=n_comp
        self.scores_std = scores.std(axis=0)  # Standardabweichung der Scores