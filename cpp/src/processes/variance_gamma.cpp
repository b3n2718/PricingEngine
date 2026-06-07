#include "processes/variance_gamma.hpp"
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mc {

VarianceGamma::VarianceGamma(const VarianceGammaParams& params) : params_(params) {
    if (params_.vol < 0)
        throw std::invalid_argument("Volatility must be non-negative");
    if (params_.spot <= 0)
        throw std::invalid_argument("Spot must be positive");
    type = "VARIANCEGAMMA";
}

py::array_t<double> VarianceGamma::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    // state: [n_sims]   — current spot prices
    // z:     [n_sims, 2]
    //   z[:,0] — standard-normal draw (Brownian component)
    //   z[:,1] — Gamma(dt/ν, ν) draw (random time-change increment Γ)
    //
    // VG log-return over [t, t+dt]:
    //   log(S(t+dt)/S(t)) = (r - q + ω)·dt + θ·Γ + σ·√Γ · Z
    //
    // where ω = (1/ν)·ln(1 - θν - σ²ν/2) is the martingale correction that
    // ensures E[S(T)] = S(0)·exp((r-q)T).
    auto s  = state.unchecked<1>();
    auto zr = z.unchecked<2>();
    int  n  = static_cast<int>(state.shape(0));

    py::array_t<double> result(n);
    auto r = result.mutable_unchecked<1>();

    // Martingale correction ω (constant for all paths and steps).
    double omega = (1.0 / params_.nu) * std::log(
        1.0 - params_.theta * params_.nu
            - 0.5 * params_.vol * params_.vol * params_.nu
    );
    double drift = (params_.risk_free_rate - params_.div_yield + omega) * dt;

    for (int i = 0; i < n; ++i) {
        double gamma_inc = zr(i, 1);  // Gamma time-change increment Γ
        double z_normal  = zr(i, 0);  // Standard-normal draw Z

        // Apply the subordinated Brownian motion:
        //   log-return = drift + θ·Γ + σ·√Γ·Z
        r(i) = s(i) * std::exp(
            drift
            + params_.theta * gamma_inc
            + params_.vol * std::sqrt(gamma_inc) * z_normal
        );
    }
    return result;
}

py::array_t<double> VarianceGamma::initial_state(int n_sims) const {
    // All paths start at the current spot price.
    py::array_t<double> state(n_sims);
    auto s = state.mutable_unchecked<1>();
    for (int i = 0; i < n_sims; ++i)
        s(i) = params_.spot;
    return state;
}

} // namespace mc
