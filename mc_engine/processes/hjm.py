from mc_engine.processes.base import StochasticProcess
from mc_engine.market.market_data import FIMarketData
from mc_engine.calibration.forward_calibration import HJMVolComponents


class HJMProcess(StochasticProcess):
    """Heath-Jarrow-Morton (HJM) forward-rate model.

    Under the HJM framework the entire forward-rate curve evolves jointly.
    The volatility structure is decomposed via PCA into n_components principal
    components, each represented as a cubic spline over the tenor grid:

        df(t, T) = μ(t, T) dt + Σ_i σ_i(T - t) dWᵢ

    The drift μ is automatically determined by the HJM no-arbitrage condition
    inside the C++ path generator.

    Parameters
    ----------
    mkt:
        Fixed-income market data containing the initial forward curve and
        tenor grid.
    vol_component:
        PCA volatility components produced by ``HJMCalibrator``.  Contains
        spline coefficients, standard deviations of scores, and tenor knots.
    """

    process_type = "HJM"

    def __init__(self, mkt: FIMarketData, vol_component: HJMVolComponents):
        self.mkt           = mkt
        self.vol           = vol_component
        self.vol_component = vol_component
        self.noise_dim     = vol_component.n_components
        # One standard-normal driver per PCA component
        self.noise = [
            {"type": "normal", "mu": 0, "sigma": 1}
            for _ in range(vol_component.n_components)
        ]

    def to_cpp_params(self) -> dict:
        return {
            "type":              "HJM",
            "r_forward":         self.mkt.r_forward.tolist(),
            "spline_parameters": self.vol_component.parameters,
            "std_scores":        self.vol_component.scores_std,
            "tenors":            self.vol_component.tenors,
            "num_vol_comp":      self.noise_dim,
            "path_type":         "forward",
        }

    def set_parameters(self, params: dict) -> None:
        pass
