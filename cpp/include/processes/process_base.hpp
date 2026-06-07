#pragma once
#include <pybind11/numpy.h>
#include <vector>

namespace mc {

namespace py = pybind11;

/**
 * @brief Abstract base class for all stochastic processes.
 *
 * Every concrete process (GBM, Heston, CIR, etc.) must inherit from this
 * class and implement the four pure-virtual methods.  The PathGenerator calls
 * these methods to run the simulation loop without knowing the concrete type.
 *
 * Design contract
 * ---------------
 * - `initial_state`  is called once before the time loop to seed the state.
 * - `evolve`         is called once per time step to advance the state by dt.
 * - `state_dim`      returns the number of state variables per simulation path
 *                    (e.g. 1 for GBM, 2 for Heston [spot, variance]).
 * - `noise_dim`      returns the number of independent Brownian drivers consumed
 *                    per time step (used to slice the global noise array).
 */
class ProcessBase {
public:
    virtual ~ProcessBase() = default;

    /**
     * @brief Advance the state by one time step.
     *
     * @param state  Current state array.  Shape depends on the process:
     *               - scalar processes (GBM, Vasicek, CIR): [n_sims]
     *               - multi-factor processes (Heston, G2++): [n_sims, state_dim]
     *               - forward-rate processes (HJM):          [n_sims, n_tenors]
     * @param z      Pre-generated noise for this time step, shape [n_sims, noise_dim].
     *               The noise has already been transformed to the required marginal
     *               distribution (normal, gamma) and correlated by the Python layer.
     * @param dt     Time step size in years.
     * @return       New state array with the same shape as `state`.
     */
    virtual py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const = 0;

    /**
     * @brief Build the initial state array for all simulation paths.
     *
     * @param n_sims  Number of Monte Carlo paths.
     * @return        Initial state array.  Shape matches the first call to `evolve`.
     */
    virtual py::array_t<double> initial_state(int n_sims) const = 0;

    /** Number of state variables per simulation path. */
    virtual int state_dim() const = 0;

    /** Number of independent noise dimensions consumed per time step. */
    virtual int noise_dim() const = 0;

    /** String tag matching the Python process_type (e.g. "GBM", "HESTON"). */
    std::string type;
};

} // namespace mc
