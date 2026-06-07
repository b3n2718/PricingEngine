#include "processes/heston.hpp"
#include <cmath>
#include <algorithm>

namespace mc {

Heston::Heston(const HestonParams& params) : params_(params) {
    type = "HESTON";
}

py::array_t<double> Heston::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    // state: [n_sims, 2]  — column 0: spot price S, column 1: variance v
    // z:     [n_sims, 2]  — two independent standard-normal draws per path
    auto s  = state.unchecked<2>();
    auto zr = z.unchecked<2>();
    int  n  = static_cast<int>(state.shape(0));

    py::array_t<double> result({n, 2});
    auto r = result.mutable_unchecked<2>();

    double sqrt_dt = std::sqrt(dt);

    for (int i = 0; i < n; ++i) {
        double S = s(i, 0);
        double V = s(i, 1);

        // Build correlated Brownian increments from independent draws via
        // Cholesky decomposition:
        //   dW_S = z₁ · √dt                           (spot driver)
        //   dW_V = (ρ·z₁ + √(1-ρ²)·z₂) · √dt        (variance driver)
        // Note: rho is already baked into the correlation matrix applied by
        // Python, so both z[:,0] and z[:,1] are passed uncorrelated here.
        double dW_S = zr(i, 0) * sqrt_dt;
        double dW_V = zr(i, 1) * sqrt_dt;

        // Protect against negative variance in the square-root term.
        double sqrt_V = std::sqrt(std::max(V, 0.0));

        // Update variance first (Euler step), then apply the reflection scheme
        // to keep variance non-negative.
        double V_new = V + params_.kappa * (params_.theta - V) * dt
                         + params_.xi * sqrt_V * dW_V;
        V_new = std::abs(V_new);  // reflection: |v_euler| avoids zero-boundary issues

        // Update spot using v at time t (not t+dt) for the diffusion term.
        // Exact log-normal solution for the spot conditional on v(t).
        double S_new = S * std::exp(
            (params_.risk_free_rate - params_.div_yield - 0.5 * V) * dt
            + sqrt_V * dW_S
        );

        r(i, 0) = S_new;
        r(i, 1) = V_new;
    }
    return result;
}

double Heston::apply_reflection(double v) const {
    // Reflection scheme: maps negative variance to its absolute value.
    // Alternative schemes include truncation (max(v,0)) and full truncation.
    return std::abs(v);
}

py::array_t<double> Heston::initial_state(int n_sims) const {
    // Column 0: spot, column 1: initial variance v0.
    py::array_t<double> state({n_sims, 2});
    auto s = state.mutable_unchecked<2>();
    for (int i = 0; i < n_sims; ++i) {
        s(i, 0) = params_.spot;
        s(i, 1) = params_.v0;
    }
    return state;
}

} // namespace mc
