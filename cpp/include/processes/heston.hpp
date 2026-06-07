#pragma once
#include "process_base.hpp"

namespace mc {

/**
 * @brief Parameters for the Heston stochastic-volatility model.
 *
 * Joint risk-neutral dynamics:
 *
 *   dS = (r - q) S dt + √v S dW₁
 *   dv = κ(θ - v) dt + ξ √v dW₂
 *   dW₁ dW₂ = ρ dt
 *
 * Correlation is applied inside `evolve` via Cholesky decomposition so that
 * the Python layer can pass independent standard-normal draws.
 */
struct HestonParams {
    double spot;           ///< Initial spot price S(0).
    double v0;             ///< Initial variance v(0) (≈ σ₀²).
    double kappa;          ///< Mean-reversion speed κ of the variance process.
    double theta;          ///< Long-run variance θ.
    double xi;             ///< Volatility of variance (vol-of-vol) ξ.
    double rho;            ///< Correlation ρ between the spot and variance drivers.
    double risk_free_rate; ///< Continuously-compounded risk-free rate r.
    double div_yield;      ///< Continuous dividend yield q.
};

/**
 * @brief Heston stochastic-volatility model.
 *
 * Two-factor model; `state_dim()` = 2, `noise_dim()` = 2.
 * The state array has shape [n_sims, 2]: column 0 = spot, column 1 = variance.
 *
 * Negative variance values produced by the Euler step are handled by the
 * reflection scheme: v_new = |v_new|.
 */
class Heston : public ProcessBase {
public:
    explicit Heston(const HestonParams& params);

    /**
     * @brief Advance all (spot, variance) pairs by one Euler-Maruyama step.
     *
     * Correlation between the two Brownian drivers is applied internally:
     *   dW_S = z[:,0] · √dt
     *   dW_V = (ρ·z[:,0] + √(1-ρ²)·z[:,1]) · √dt
     *
     * @note The Python Correlator also applies correlation via Cholesky before
     *       passing `z` here.  For Heston the Python layer passes the two noise
     *       dimensions uncorrelated; the correlation is handled here.
     *
     * @param state  Current state [n_sims, 2]: col 0 = spot, col 1 = variance.
     * @param z      Independent standard-normal draws [n_sims, 2].
     * @param dt     Time step size in years.
     * @return       New state [n_sims, 2].
     */
    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    int state_dim() const override { return 2; }
    int noise_dim() const override { return 2; }

    /** Build initial state: col 0 = spot, col 1 = v0, shape [n_sims, 2]. */
    py::array_t<double> initial_state(int n_sims) const override;

private:
    HestonParams params_;

    /** Reflection scheme: maps negative variance to its absolute value. */
    double apply_reflection(double v) const;
};

} // namespace mc
