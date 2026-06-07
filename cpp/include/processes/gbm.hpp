#pragma once
#include "process_base.hpp"

namespace mc {

/**
 * @brief Parameters for the Geometric Brownian Motion process.
 *
 * Under the risk-neutral measure the spot price evolves as:
 *
 *   dS = (r - q) S dt + σ S dW
 *
 * The Euler-Maruyama step uses the exact log-normal solution to avoid
 * discretisation error:
 *
 *   S(t+dt) = S(t) · exp((r - q - σ²/2) dt + σ √dt · Z)
 */
struct GBMParams {
    double spot;           ///< Initial spot price S(0).
    double vol;            ///< Constant annualised volatility σ.
    double risk_free_rate; ///< Continuously-compounded risk-free rate r.
    double div_yield;      ///< Continuous dividend yield q.
};

/**
 * @brief Geometric Brownian Motion equity process.
 *
 * Single-factor model; `state_dim()` = 1, `noise_dim()` = 1.
 * The state array has shape [n_sims] and stores the current spot price.
 */
class GBM : public ProcessBase {
public:
    explicit GBM(const GBMParams& params);

    /**
     * @brief Advance all spot prices by one time step using the exact GBM solution.
     *
     * @param state  Current spot prices, shape [n_sims].
     * @param z      Standard-normal draws, shape [n_sims, 1].
     * @param dt     Time step size in years.
     * @return       New spot prices, shape [n_sims].
     */
    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    int state_dim() const override { return 1; }
    int noise_dim() const override { return 1; }

    /** Build initial state: all paths start at params_.spot. */
    py::array_t<double> initial_state(int n_sims) const override;

private:
    GBMParams params_;
};

} // namespace mc
