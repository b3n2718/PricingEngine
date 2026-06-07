#pragma once
#include "process_base.hpp"

namespace mc {

/**
 * @brief Parameters for the Variance-Gamma jump-diffusion process.
 *
 * The log-price increment over [t, t+dt] is:
 *
 *   log(S(t+dt)/S(t)) = (r - q + ω) dt + θ·Γ + σ·√Γ · Z
 *
 * where Γ ~ Gamma(dt/ν, ν) is the stochastic time-change increment and
 * ω = (1/ν)·ln(1 - θν - σ²ν/2) is the martingale correction.
 *
 * Two noise channels are consumed per step: Z (standard normal) and Γ (gamma).
 * The gamma draws are pre-generated in Python with shape parameters a = dt/ν
 * and b = ν (so the mean is dt and variance is νdt), and passed as z[:,1].
 */
struct VarianceGammaParams {
    double spot;           ///< Initial spot price S(0).
    double vol;            ///< Volatility σ of the Brownian component.
    double risk_free_rate; ///< Continuously-compounded risk-free rate r.
    double div_yield;      ///< Continuous dividend yield q.
    double theta;          ///< Drift θ of the Brownian motion in the time-change (controls skewness).
    double nu;             ///< Variance ν of the Gamma time-change (controls kurtosis).
};

/**
 * @brief Variance-Gamma equity process.
 *
 * `state_dim()` = 1, `noise_dim()` = 2.
 * The state array has shape [n_sims] and stores the current spot price.
 * z[:,0] is the standard-normal draw; z[:,1] is the Gamma time-change increment.
 */
class VarianceGamma : public ProcessBase {
public:
    explicit VarianceGamma(const VarianceGammaParams& params);

    /**
     * @brief Advance all spot prices by one VG step.
     *
     * @param state  Current spot prices, shape [n_sims].
     * @param z      Noise matrix [n_sims, 2]: col 0 = N(0,1), col 1 = Gamma(dt/ν, ν).
     * @param dt     Time step size in years.
     * @return       New spot prices, shape [n_sims].
     */
    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    int state_dim() const override { return 1; }
    int noise_dim() const override { return 2; }

    /** Build initial state: all paths start at params_.spot. */
    py::array_t<double> initial_state(int n_sims) const override;

private:
    VarianceGammaParams params_;
};

} // namespace mc
