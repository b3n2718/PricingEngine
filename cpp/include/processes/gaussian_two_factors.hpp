#pragma once
#include "process_base.hpp"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace mc {

namespace py = pybind11;

/**
 * @brief Parameters for the G2++ two-factor Hull-White model.
 *
 * The short rate decomposes into two mean-reverting Gaussian factors plus a
 * deterministic shift φ(t) that fits the initial yield curve:
 *
 *   r(t) = x(t) + y(t) + φ(t)
 *   dx   = -a·x dt + σ dW₁
 *   dy   = -b·y dt + η dW₂
 *   dW₁ dW₂ = ρ dt
 *
 * The Euler-Maruyama step for the two factors is:
 *
 *   x(t+dt) = x(t) - a·x(t) dt + σ dW_x
 *   y(t+dt) = y(t) - b·y(t) dt + η dW_y
 *
 * Correlation is applied inside `evolve` via a two-dimensional Cholesky factor:
 *   dW_x = z₁ √dt
 *   dW_y = (ρ·z₁ + √(1-ρ²)·z₂) √dt
 *
 * The shift φ(t) is computed analytically in the Python PathData container
 * (G2ppPath) rather than in the C++ step, keeping the state minimal.
 */
struct G2ppParams {
    double a;     ///< Mean-reversion speed of the first factor x(t).
    double b;     ///< Mean-reversion speed of the second factor y(t).
    double sigma; ///< Volatility σ of the first factor.
    double eta;   ///< Volatility η of the second factor.
    double rho;   ///< Instantaneous correlation ρ between the two Brownian drivers.
    double x0;    ///< Initial value of x (typically 0).
    double y0;    ///< Initial value of y (typically 0).
};

/**
 * @brief G2++ two-factor Hull-White short-rate process.
 *
 * Two-factor model; `state_dim()` = 2, `noise_dim()` = 2.
 * The state array has shape [n_sims, 2]: column 0 = x(t), column 1 = y(t).
 *
 * Bond prices P(t1, t2) are computed analytically in the Python layer using
 * the closed-form G2++ formula, so no nested simulation is required.
 */
class G2pp : public ProcessBase {
public:
    explicit G2pp(const G2ppParams& params);

    /**
     * @brief Advance both factors by one Euler-Maruyama step.
     *
     * Internally applies the Cholesky decomposition for ρ so that
     * the two noise channels can be passed as independent N(0,1) draws.
     *
     * @param state  Current factor values [n_sims, 2]: col 0 = x, col 1 = y.
     * @param z      Independent standard-normal draws [n_sims, 2].
     * @param dt     Time step size in years.
     * @return       New factor values [n_sims, 2].
     */
    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    /** Build initial state: col 0 = x0, col 1 = y0, shape [n_sims, 2]. */
    py::array_t<double> initial_state(int n_sims) const override;

    int state_dim() const override { return 2; }
    int noise_dim() const override { return 2; }

private:
    G2ppParams params_;
};

} // namespace mc
