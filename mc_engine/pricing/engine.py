import numpy as np
from pricing.config import MCConfig
from pricing.result import PricingResult
from processes.base import StochasticProcess
from products.base import Product
from paths.path_factory import build_path
from simulation.rng import RandomNumberGenerator
from simulation.correlator import Correlator
from market.curves import YieldCurve

import mc_engine.mc_core as mc_core

class MonteCarloEngine:

    def __init__(self, config: MCConfig = MCConfig()):
        self.config     = config
        self.rng        = RandomNumberGenerator(config.rng_type, config.seed)
        self.correlator = Correlator()

    def price(self,
              product:   Product,
              processes: dict[str, StochasticProcess],
              curve:     YieldCurve) -> PricingResult:

        cfg = self.config
        dt  = product.maturity / cfg.n_steps
        df  = curve.discount(product.maturity)

        asset_ids        = product.underlyings

        params_list      = [processes[a].to_cpp_params() for a in asset_ids]
        total_noise_dim  = sum(processes[a].noise_dim for a in asset_ids)

        # 1. Zufallszahlen in Python generieren
        n = cfg.n_sims // 2 if cfg.use_antithetic else cfg.n_sims

        z = list()
        total_noise_dim = 0
        for i,a in enumerate(asset_ids):
            processes[a].set_parameters({"dt":dt})
            z.append(self.rng.generate(n, cfg.n_steps, processes[a].noise))
        
            # 2. Antithetic Variates verdoppeln
            if cfg.use_antithetic:
                z[i] = np.concatenate([z[i], -z[i]], axis=0)   # [n_sims, n_steps, noise_dim]
            total_noise_dim += processes[a].noise_dim
        z = np.concat(z,axis=2)#.reshape(n, cfg.n_steps,total_noise_dim)
        # 3. Für Single-Asset: keine Korrelation nötig
        # (bei Multi-Asset kommt hier Cholesky rein)
        
        if total_noise_dim > 1 and self.config.corrolation_matrix is None:        
            chol = np.eye(total_noise_dim)
            z    = self.correlator.correlate(z, chol)
        elif total_noise_dim > 1 and self.config.corrolation_matrix is not None:
            z    = self.correlator.correlate(z, self.config.corrolation_matrix)

        # 4. Pfade in C++ simulieren
        paths_raw: dict = mc_core.generate(params_list, asset_ids, z, dt)

        # 5. Rohe Arrays in typisierte PathData Objekte wandeln
        model_params = dict()
        for asset_id in asset_ids:
            model_params[asset_id] = processes[asset_id].to_cpp_params()
            model_params[asset_id].update({"dt":dt})
        
        self.paths = {
            asset_id: build_path(paths_raw[asset_id], processes[asset_id].process_type, model_params[asset_id])
            for asset_id in asset_ids
        }

        # 6. Payoffs berechnen
        payoffs = product.payoff(self.paths, df)
        return self._compute_result(payoffs) 

    def _compute_result(self, payoffs: np.ndarray) -> PricingResult:
        n  = len(payoffs)
        mu = float(payoffs.mean())
        se = float(payoffs.std() / np.sqrt(n))
        return PricingResult(
            price     = mu,
            std_error = se,
            ci_lower  = mu - 1.96 * se,
            ci_upper  = mu + 1.96 * se,
        )