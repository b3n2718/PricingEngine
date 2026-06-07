# PricingEngine

A Monte Carlo derivatives pricing library that combines a high-performance C++ simulation core (via [pybind11](https://github.com/pybind/pybind11)) with Python-level calibration, product definitions, and market data handling.

## Features

- **8 stochastic processes** — equity and fixed-income models
- **8 derivative products** — equity options and interest-rate instruments
- **Fourier-based calibration** for Heston and Variance-Gamma models
- **PCA-based HJM calibration** from historical forward-rate data
- **Analytic G2++ calibration** to ATM swaption volatilities
- **MLE short-rate calibration** (Vasicek and CIR)
- **Variance reduction** via antithetic variates and quasi-random (Sobol) sampling
- **Multi-asset correlation** via Cholesky decomposition

---

## Supported Models

| Model | Type | Key Parameters |
|---|---|---|
| **GBM** | Equity | σ, r, q |
| **Heston** | Equity (Stochastic Vol) | v₀, κ, θ, ξ, ρ |
| **Variance-Gamma** | Equity (Jump-Diffusion) | σ, θ, ν |
| **Vasicek** | Short Rate | κ, θ, σ |
| **CIR** | Short Rate | κ, θ, σ |
| **HJM** | Forward Rate (PCA) | PCA vol components |
| **G2++** | Two-Factor Short Rate | a, b, σ, η, ρ |

---

## Supported Products

### Equity
| Product | Description |
|---|---|
| `EuropeanOption` | Standard call/put — payoff at maturity |
| `AmericanOption` | Early-exercise option via Longstaff-Schwartz |
| `AsianOption` | Arithmetic or geometric average-price option |
| `BasketOption` | Weighted basket of multiple underlyings |

### Fixed Income
| Product | Description |
|---|---|
| `ZeroCouponBond` | Discounted notional using stochastic discount factor |
| `BondOption` | European call on a zero-coupon bond |
| `Swaption` | Payer swaption on a fixed-for-floating swap |
| `Caplet` | Call on a single LIBOR period |

---

## Installation

### Prerequisites

- Python ≥ 3.11
- A C++17 compiler (GCC / Clang / MSVC)
- `pybind11`

### Build the C++ extension

```bash
pip install pybind11
pip install -r requirements.txt
pip install -e .
```

The `setup.py` compiles the C++ path generator into `mc_engine.mc_core`.  All Python imports then work normally:

```python
from mc_engine.pricing.engine import MonteCarloEngine
```

---

## Quick Start

### European option with GBM

```python
import numpy as np
from mc_engine.market.curves import YieldCurve
from mc_engine.market.market_data import EquityMarketData
from mc_engine.processes.gbm import GBMProcess
from mc_engine.products.equity.european import EuropeanOption
from mc_engine.pricing.engine import MonteCarloEngine
from mc_engine.pricing.config import MCConfig

# Market data
curve = YieldCurve(tenors=np.array([0.5, 1, 2, 5]), rates=np.array([0.04, 0.045, 0.05, 0.055]))
mkt   = EquityMarketData(spot=100.0, div_yield=0.02, curve=curve)

# Process and product
process = GBMProcess(mkt, vol=0.20)
option  = EuropeanOption("SPX", strike=105.0, maturity=1.0, is_call=True)

# Price
engine = MonteCarloEngine(MCConfig(n_sims=100_000, n_steps=252))
result = engine.price(option, {"SPX": process}, curve)

print(f"Price:  {result.price:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

### Heston model with calibration

```python
from mc_engine.calibration.heston_calibration import HestonCalibrator
from mc_engine.market.vol_surface import VolSurface
from mc_engine.processes.heston import HESTONProcess

# Build vol surface from market data
surface = VolSurface(strikes, maturities, market_vols)

# Calibrate Heston parameters
calibrator = HestonCalibrator(S=100.0, r=0.05, q=0.0, vol_surface=surface)
params     = calibrator.calibrate()

# Price with calibrated process
process = HESTONProcess(mkt, **params)
result  = engine.price(option, {"SPX": process}, curve)
```

### Interest-rate product with CIR

```python
from mc_engine.market.market_data import FIMarketData
from mc_engine.processes.cir import CIRProcess
from mc_engine.products.fixed_income.zcb import ZeroCouponBond

fi_mkt  = FIMarketData(r_spot=0.04, r_forward=fwd_rates, forward_tenors=tenors)
process = CIRProcess(fi_mkt, vol=0.01, kappa=0.5, theta=0.04)
bond    = ZeroCouponBond("RATE", notional=1000.0, maturity=5.0)

result  = engine.price(bond, {"RATE": process}, curve)
```

---

## Project Structure

```
PricingEngine/
├── cpp/                            # C++ path generator (pybind11 extension)
│   ├── bindings/bindings.cpp       # Python bindings
│   ├── include/                    # Header files
│   │   ├── path_generator.hpp
│   │   ├── processes/              # Process headers (GBM, Heston, ...)
│   │   └── utils/cubic_splines.hpp
│   └── src/                        # Implementation files
│       ├── path_generator.cpp
│       ├── processes/
│       └── utils/
├── mc_engine/                      # Python package
│   ├── calibration/                # Calibration routines
│   │   ├── base.py                 # Abstract Calibrator (L-BFGS-B)
│   │   ├── fourier.py              # Gil-Pelaez Fourier pricer
│   │   ├── implied_vol.py          # Black-Scholes pricing and IV solver
│   │   ├── heston_calibration.py   # Heston CF calibration
│   │   ├── variance_gamma_calbration.py
│   │   ├── short_rate_calibration.py   # Vasicek / CIR MLE
│   │   ├── forward_calibration.py      # HJM PCA calibration
│   │   └── g2pp_calibration.py         # G2++ swaption calibration
│   ├── market/                     # Market data containers
│   │   ├── curves.py               # YieldCurve
│   │   ├── market_data.py          # EquityMarketData, FIMarketData
│   │   └── vol_surface.py          # VolSurface (bicubic spline)
│   ├── paths/                      # PathData containers
│   │   ├── base.py                 # Abstract PathData
│   │   ├── scalar_path.py          # GBM / Heston / VG paths
│   │   ├── spot_rate_model.py      # Vasicek / CIR (with analytic ZCP)
│   │   ├── forward_rate_model.py   # HJM forward curves
│   │   ├── gaussian_two_factors_path.py  # G2++ (with analytic ZCP)
│   │   └── path_factory.py         # Factory: raw array → PathData
│   ├── pricing/                    # Engine and configuration
│   │   ├── engine.py               # MonteCarloEngine
│   │   ├── config.py               # MCConfig dataclass
│   │   └── result.py               # PricingResult dataclass
│   ├── processes/                  # Stochastic process definitions
│   │   ├── base.py                 # Abstract StochasticProcess
│   │   ├── gbm.py                  # Geometric Brownian Motion
│   │   ├── heston.py               # Heston stochastic volatility
│   │   ├── vasicek.py              # Vasicek short rate
│   │   ├── cir.py                  # Cox-Ingersoll-Ross
│   │   ├── hjm.py                  # Heath-Jarrow-Morton
│   │   ├── variance_gamma.py       # Variance-Gamma
│   │   └── gaussian_two_factors.py # G2++ two-factor Hull-White
│   ├── products/                   # Derivative payoff definitions
│   │   ├── base.py                 # Abstract Product
│   │   ├── equity/
│   │   │   ├── european.py
│   │   │   ├── american.py         # Longstaff-Schwartz LSM
│   │   │   ├── asian.py
│   │   │   └── basket.py
│   │   └── fixed_income/
│   │       ├── zcb.py
│   │       ├── bond_option.py
│   │       ├── swaption.py
│   │       └── caplet.py
│   └── simulation/                 # Random number generation
│       ├── rng.py                  # Pseudo / Sobol with inverse-CDF transforms
│       └── correlator.py           # Cholesky correlation
├── examples/
│   └── data/hjm.csv                # Sample forward-rate data
├── requirements.txt
├── setup.py
└── pyproject.toml
```

---

## Architecture Overview

```
MonteCarloEngine.price(product, processes, curve)
         │
         ├── RandomNumberGenerator.generate()    → z [n_sims, n_steps, noise_dim]
         ├── Correlator.correlate(z, chol)        → correlated z
         ├── mc_core.generate(params, ids, z, dt) → raw paths (C++)
         ├── build_path(raw, type, params)         → PathData objects
         └── product.payoff(paths, discount)       → payoffs [n_sims]
                                                    → PricingResult
```

### Calibration Flow

```
Market Data (VolSurface / Rate Series)
         │
         ▼
    Calibrator.calibrate()          ← L-BFGS-B optimisation
         │
         ▼
    Calibrated Process Parameters
         │
         ▼
    StochasticProcess(mkt, **params)
         │
         ▼
    MonteCarloEngine.price(...)
```

---

## Configuration

`MCConfig` controls simulation behaviour:

```python
from mc_engine.pricing.config import MCConfig
from mc_engine.simulation.rng import RNGType

config = MCConfig(
    n_sims             = 100_000,       # number of paths
    n_steps            = 252,           # daily steps for 1-year option
    rng_type           = RNGType.SOBOL, # quasi-random for lower variance
    seed               = 42,
    use_antithetic     = True,          # mirrors half the paths
    corrolation_matrix = chol,          # Cholesky factor (multi-asset)
)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `scipy` | Optimisation, quadrature, statistics |
| `pandas` | Forward-rate data handling (HJM) |
| `scikit-learn` | PCA for HJM calibration |
| `pybind11` | C++ Python bindings |
