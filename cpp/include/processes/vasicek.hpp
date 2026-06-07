#pragma once
#include "process_base.hpp"

namespace mc {

/**
 * @brief Parameters for the Vasicek short-rate model.
 *
 * The short rate r(t) evolves as:
 *
 *   dr = κ(θ - r) dt + σ dW
 *
 * The Euler-Maruyama discretisation is:
 *
 *   r(t+dt) = r(t) + κ(θ - r(t)) dt + σ √dt · Z
 *
 * The model allows negative rates and has Gaussian transition densities,
 * enabling exact maximum-likelihood calibration.
 */
struct VasicekParams {
    double r_spot; ///< Initial short rate r(0).
    double kappa;  ///< Mean-reversion speed κ.
    double theta;  ///< Long-run mean θ of the short rate.
    double vol;    ///< Constant short-rate volatility σ.
};

/**
 * @brief Vasicek mean-reverting short-rate process.
 *
 * Single-factor model; `state_dim()` = 1, `noise_dim()` = 1.
 * The state array has shape [n_sims] and stores the current short rate.
 */
class Vasicek : public ProcessBase {
public:
    explicit Vasicek(const VasicekParams& params);

    /**
     * @brief Advance all short rates by one Euler-Maruyama step.
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
    VasicekParams params_;
};

} // namespace mc
