#pragma once
#include "processes/process_base.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <memory>

namespace mc {

/**
 * @brief Orchestrates Monte Carlo path simulation for one or more assets.
 *
 * All public methods are static; the class acts as a namespace for the
 * simulation pipeline.  The main entry point is `generate`, which is exposed
 * to Python via pybind11.
 *
 * Pipeline
 * --------
 * 1. For each asset, `build_process` constructs the concrete ProcessBase from
 *    the parameter dict passed from Python.
 * 2. The noise columns belonging to that asset are sliced out of the global
 *    noise tensor `z` (which covers all assets).
 * 3. `simulate_process_spot` or `simulate_process_forward` runs the time loop,
 *    calling `process.evolve` at each step and recording the output.
 * 4. Results are returned as a Python dict mapping asset_id → ndarray.
 */
class PathGenerator {
public:

    /**
     * @brief Main entry point: simulate paths for all assets and return a dict.
     *
     * @param params_list  One parameter dict per asset (from Python's
     *                     `StochasticProcess.to_cpp_params()`).
     * @param asset_ids    Asset identifier strings; used as dict keys in the output.
     * @param z            Pre-generated, correlated noise tensor
     *                     [n_sims, n_steps, total_noise_dim].  Each asset's noise
     *                     channels are a contiguous slice along axis 2.
     * @param dt           Time step size in years.
     * @return             Python dict { asset_id → ndarray }.
     *                     - Spot processes (GBM, Heston, VG, Vasicek, CIR):
     *                       shape [n_sims, n_steps].
     *                     - G2++: shape [n_sims, 2, n_steps] (factors x and y).
     *                     - HJM:  shape [n_sims, n_tenors, n_steps].
     */
    static pybind11::dict generate(
        const std::vector<pybind11::dict>& params_list,
        const std::vector<std::string>&    asset_ids,
        const pybind11::array_t<double>&   z,
        double dt
    );

private:
    /**
     * @brief Factory: construct the correct ProcessBase subclass from a param dict.
     *
     * Reads the "type" key and dispatches to the appropriate constructor.
     * Throws std::invalid_argument for unknown process types.
     */
    static std::unique_ptr<ProcessBase> build_process(
        const pybind11::dict& params
    );

    /**
     * @brief Time loop for scalar and two-factor spot-type processes.
     *
     * Calls `process.initial_state`, then loops over time steps calling
     * `process.evolve` and writing the relevant state component(s) into the
     * output array.
     *
     * Output shape:
     * - state_dim == 1 (GBM, Vasicek, CIR, VG): [n_sims, n_steps]
     * - type == "G2PP":                           [n_sims, 2, n_steps]
     * - type == "HESTON":                         [n_sims, n_steps] (spot only)
     *
     * @param process  Constructed process to evolve.
     * @param z_asset  Noise slice for this asset [n_sims, n_steps, noise_dim].
     * @param dt       Time step size in years.
     * @param n_sims   Number of Monte Carlo paths.
     * @param n_steps  Number of time steps.
     * @return         Output path array.
     */
    static pybind11::array_t<double> simulate_process_spot(
        const ProcessBase&               process,
        const pybind11::array_t<double>& z_asset,
        double dt,
        int n_sims,
        int n_steps
    );

    /**
     * @brief Time loop for forward-curve processes (HJM).
     *
     * The full forward curve is recorded at every time step.
     * Output shape: [n_sims, n_tenors, n_steps].
     *
     * @param process  Constructed HJM process.
     * @param z_asset  Noise slice [n_sims, n_steps, n_factors].
     * @param dt       Time step size in years.
     * @param n_sims   Number of Monte Carlo paths.
     * @param n_steps  Number of time steps.
     * @param n_tenors Number of forward-curve tenor grid points.
     * @return         Forward-curve path array [n_sims, n_tenors, n_steps].
     */
    static pybind11::array_t<double> simulate_process_forward(
        const ProcessBase&               process,
        const pybind11::array_t<double>& z_asset,
        double dt,
        int n_sims,
        int n_steps,
        int n_tenors
    );
};

} // namespace mc
