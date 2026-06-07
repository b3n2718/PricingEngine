#pragma once
#include "process_base.hpp"

namespace mc {

/**
 * @brief Parameters for the Cox-Ingersoll-Ross short-rate model.
 *
 * The short rate r(t) evolves as:
 *
 *   dr = κ(θ - r) dt + σ √r dW
 *
 * The square-root diffusion coefficient keeps rates non-negative when the
 * Feller condition 2κθ > σ² is satisfied.  The Euler step is:
 *
 *   r(t+dt) = r(t) + κ(θ - r(t)) dt + σ √(max(r(t),0)) √dt · Z
 */
struct CIRParams {
    double r_spot; ///< Initial short rate r(0).
    double kappa;  ///< Mean-reversion speed κ.
    double theta;  ///< Long-run mean θ of the short rate.
    double vol;    ///< Diffusion coefficient σ.
};

/**
 * @brief Cox-Ingersoll-Ross short-rate process.
 *
 * Single-factor model; `state_dim()` = 1, `noise_dim()` = 1.
 * The state array has shape [n_sims] and stores the current short rate.
 */
class CIR : public ProcessBase {
public:
    explicit CIR(const CIRParams& params);

    /**
     * @brief Advance all short rates by one Euler-Maruyama step.
     *
     * The square root is evaluated at max(r(t), 0) to handle rare cases
     * where the Euler step produces a slightly negative rate.
     *
     * @param state  Current short rates, shape [n_sims].
     * @param z      Standard-normal draws, shape [n_sims, 1].
     * @param dt     Time step size in years.
     * @return       New short rates, shape [n_sims].
     */
    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    int state_dim() const override { return 1; }
    int noise_dim() const override { return 1; }

    /** Build initial state: all paths start at params_.r_spot. */
    py::array_t<double> initial_state(int n_sims) const override;

private:
    CIRParams params_;
};

} // namespace mc
