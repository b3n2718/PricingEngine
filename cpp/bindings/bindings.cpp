#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "path_generator.hpp"

namespace py = pybind11;

/**
 * @brief pybind11 module definition for the C++ Monte Carlo core.
 *
 * Exposes `mc::PathGenerator::generate` as `mc_core.generate` so the Python
 * engine can call the high-performance path simulation directly:
 *
 *   import mc_engine.mc_core as mc_core
 *   paths = mc_core.generate(params_list, asset_ids, z, dt)
 */
PYBIND11_MODULE(mc_core, m) {
    m.doc() = "Monte Carlo path-generation core (C++17, pybind11)";

    m.def("generate",
          &mc::PathGenerator::generate,
          py::arg("params_list"),
          py::arg("asset_ids"),
          py::arg("z"),
          py::arg("dt"),
          R"doc(
Simulate price/rate paths for one or more assets given pre-generated noise.

Parameters
----------
params_list : list[dict]
    One parameter dict per asset.  Each dict must contain at least "type"
    and "path_type".  Additional keys depend on the process (e.g. "spot",
    "vol" for GBM; "v0", "kappa", "theta", "xi", "rho" for Heston).
asset_ids : list[str]
    Asset identifier strings.  Used as keys in the returned dict.
    Must have the same length as params_list.
z : ndarray, shape [n_sims, n_steps, total_noise_dim]
    Pre-generated, optionally correlated noise.  Each process consumes
    `noise_dim()` columns from axis 2 in the order assets appear in asset_ids.
dt : float
    Simulation time step in years (maturity / n_steps).

Returns
-------
dict[str, ndarray]
    Mapping from asset_id to the simulated path array.  Shape varies by process:
    - Spot processes (GBM, Vasicek, CIR, VG, Heston): [n_sims, n_steps]
    - G2++: [n_sims, 2, n_steps]  (factors x and y)
    - HJM:  [n_sims, n_tenors, n_steps]
          )doc"
    );
}
