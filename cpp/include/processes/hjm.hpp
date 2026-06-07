// hjm.hpp
#pragma once
#include "process_base.hpp"
#include "utils/cubic_splines.hpp"

namespace mc {

/**
 * @brief Parameters for the Heath-Jarrow-Morton forward-rate model.
 *
 * Under the HJM no-arbitrage framework the entire forward curve evolves
 * jointly.  The volatility structure is represented as a PCA decomposition
 * with n_factors components, each described by a cubic spline over the
 * tenor grid.
 *
 * The discretised SDE for each tenor T_j is:
 *
 *   f(t+dt, T_j) = f(t, T_j) + α_j dt + Σ_k σ_k(T_j) √dt · Z_k
 *
 * where α_j is the HJM no-arbitrage drift computed once in the constructor.
 */
struct HJMParams {
    const py::array_t<double> r_forward;        ///< Initial forward-rate curve, shape [n_tenors].
    const py::array_t<double> std_scores;        ///< PCA score standard deviations, shape [n_factors].
    const py::array_t<double> spline_parameters; ///< Spline coefficients for all components, shape [n_factors, n_segments, 4].
    const py::array_t<double> tenors;            ///< Tenor grid knots in years, shape [n_tenors].
    int                       n_factors;         ///< Number of retained PCA components.
};

/**
 * @brief Heath-Jarrow-Morton forward-rate process.
 *
 * Multi-factor model; `state_dim()` = n_tenors, `noise_dim()` = n_factors.
 * The state array has shape [n_sims, n_tenors] and stores the full forward
 * curve at each time step.
 *
 * Internally the constructor builds one CubicSpline per PCA component and
 * pre-computes the HJM drift vector α so that the time-step loop is efficient.
 */
class HJM : public ProcessBase {
public:
    explicit HJM(const HJMParams& params);

    /**
     * @brief Advance the full forward curve by one Euler step.
     *
     * @param state  Current forward curves, shape [n_sims, n_tenors].
     * @param z      Standard-normal draws, shape [n_sims, n_factors].
     * @param dt     Time step size in years.
     * @return       New forward curves, shape [n_sims, n_tenors].
     */
    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    /** n_tenors — full forward curve is the state. */
    int state_dim() const override { return static_cast<int>(params_.tenors.size()); }
    int noise_dim() const override { return params_.n_factors; }

    /** Build initial state: all paths start with the market forward curve. */
    py::array_t<double> initial_state(int n_sims) const override;

private:
    HJMParams              params_;
    std::vector<CubicSpline> splines_; ///< One spline per PCA component, built in the constructor.
    std::vector<double>    alpha_;     ///< Pre-computed HJM no-arbitrage drift for each tenor.

    /** Build a CubicSpline for each PCA volatility component. */
    void build_splines();

    /**
     * @brief Evaluate the k-th volatility component at tenor τ.
     *
     * Returns spline_k(τ) · std_scores[k], scaling the normalised PCA
     * eigenvector back to historical volatility units.
     */
    double vol(int k, double tau) const;

    /**
     * @brief Pre-compute the HJM no-arbitrage drift α_j for each tenor T_j.
     *
     * The HJM drift condition is:
     *
     *   α(t, T) = Σ_k σ_k(T) ∫_0^T σ_k(s) ds
     *
     * This is computed once at construction time and reused in every `evolve` call.
     */
    void compute_alpha();
};

} // namespace mc
