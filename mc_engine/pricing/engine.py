import numpy as np
from mc_engine.pricing.config import MCConfig
from mc_engine.pricing.result import PricingResult
from mc_engine.processes.base import StochasticProcess
from mc_engine.products.base import Product
from mc_engine.paths.path_factory import build_path
from mc_engine.simulation.rng import RandomNumberGenerator
from mc_engine.simulation.correlator import Correlator
from mc_engine.market.curves import YieldCurve

import mc_engine.mc_core as mc_core


class MonteCarloEngine:
    """Main Monte Carlo pricing engine.

    Orchestrates the full simulation pipeline:

    1. Determine the time step ``dt = maturity / n_steps``.
    2. Generate random numbers (pseudo or quasi) for each asset.
    3. Optionally apply antithetic variates (mirror the draws).
    4. Apply Cholesky correlation if required.
    5. Delegate path simulation to the C++ backend (``mc_core.generate``).
    6. Wrap raw arrays in typed ``PathData`` containers.
    7. Compute payoffs via the product's ``payoff`` method.
    8. Return the mean price and 95 % confidence interval.

    Parameters
    ----------
    config:
        Simulation configuration (paths, steps, RNG type, antithetic flag).
    """

    def __init__(self, config: MCConfig = MCConfig()):
        self.config     = config
        self.rng        = RandomNumberGenerator(config.rng_type, config.seed)
        self.correlator = Correlator()

    def price(self,
              product:   Product,
              processes: dict[str, StochasticProcess],
              curve:     YieldCurve) -> PricingResult:
        """Price a derivative product via Monte Carlo simulation.

        Parameters
        ----------
        product:
            The derivative instrument defining the payoff and maturity.
        processes:
            Mapping from asset identifier to its stochastic process.  Keys
            must match ``product.underlyings``.
        curve:
            Risk-free yield curve for the deterministic discount factor.

        Returns
        -------
        PricingResult
            Price estimate together with standard error and 95 % CI.
        """
        cfg = self.config
        dt  = product.maturity / cfg.n_steps
        df  = curve.discount(product.maturity)

        asset_ids   = product.underlyings
        params_list = [processes[a].to_cpp_params() for a in asset_ids]

        # Half paths when antithetic variates are used; they are mirrored below
        n = cfg.n_sims // 2 if cfg.use_antithetic else cfg.n_sims

        # Generate random draws per asset and concatenate along the noise axis
        z               = []
        total_noise_dim = 0
        for i, a in enumerate(asset_ids):
            processes[a].set_parameters({"dt": dt})
            z.append(self.rng.generate(n, cfg.n_steps, processes[a].noise))

            if cfg.use_antithetic:
                # Append the mirror of each draw to double the path count
                z[i] = np.concatenate([z[i], -z[i]], axis=0)

            total_noise_dim += processes[a].noise_dim

        z = np.concat(z, axis=2)  # [n_sims, n_steps, total_noise_dim]

        # Apply inter-asset correlation via Cholesky decomposition
        if total_noise_dim > 1 and self.config.corrolation_matrix is None:
            chol = np.eye(total_noise_dim)
            z    = self.correlator.correlate(z, chol)
        elif total_noise_dim > 1 and self.config.corrolation_matrix is not None:
            z    = self.correlator.correlate(z, self.config.corrolation_matrix)

        # Delegate to the C++ path generator
        paths_raw: dict = mc_core.generate(params_list, asset_ids, z, dt)
        self.raw_data   = paths_raw

        # Augment each process's params with dt and wrap in typed containers
        model_params = {}
        for asset_id in asset_ids:
            model_params[asset_id] = processes[asset_id].to_cpp_params()
            model_params[asset_id]["dt"] = dt

        self.paths = {
            asset_id: build_path(
                paths_raw[asset_id],
                processes[asset_id].process_type,
                model_params[asset_id],
            )
            for asset_id in asset_ids
        }

        payoffs = product.payoff(self.paths, df)
        return self._compute_result(payoffs)

    def _compute_result(self, payoffs: np.ndarray) -> PricingResult:
        """Compute price statistics from an array of discounted payoffs."""
        n  = len(payoffs)
        mu = float(payoffs.mean())
        se = float(payoffs.std() / np.sqrt(n))
        return PricingResult(
            price     = mu,
            std_error = se,
            ci_lower  = mu - 1.96 * se,
            ci_upper  = mu + 1.96 * se,
        )
