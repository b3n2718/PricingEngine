#include "processes/heston.hpp"
#include <cmath>
#include <algorithm>

namespace mc {

Heston::Heston(const HestonParams& params) : params_(params) {}

py::array_t<double> Heston::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    // state: [n_sims, 2] — Spalte 0: Spot, Spalte 1: Varianz
    // z:     [n_sims, 2] — zwei unabhängige Zufallszahlen
    auto s  = state.unchecked<2>();
    auto zr = z.unchecked<2>();
    int  n  = state.shape(0);

    py::array_t<double> result({n, 2});
    auto r = result.mutable_unchecked<2>();

    double sqrt_dt = std::sqrt(dt);

    for (int i = 0; i < n; ++i) {
        double S = s(i, 0);
        double V = s(i, 1);

        // Korrelierte Brownschen Bewegungen aufbauen
        // z(:,0) → dW^S, z(:,1) → dW^V (unabhängig)
        // rho bereits durch Cholesky in Python angewendet
        double dW_S = zr(i, 0) * sqrt_dt;
        double dW_V = zr(i, 1) * sqrt_dt;

        double sqrt_V = std::sqrt(std::max(V, 0.0));

        // Varianz zuerst updaten (Euler)
        double V_new = V + params_.kappa * (params_.theta - V) * dt
                         + params_.xi * sqrt_V * dW_V;

        // Reflection scheme: negative Varianz spiegeln
        V_new = std::abs(V_new);

        // Spot updaten mit V zum Zeitpunkt t (nicht t+dt)
        double S_new = S * std::exp(
            (params_.risk_free_rate - params_.div_yield
             - 0.5 * V) * dt
            + sqrt_V * dW_S
        );

        r(i, 0) = S_new;
        r(i, 1) = V_new;
    }
    return result;
}

double Heston::apply_reflection(double v) const {
    return std::abs(v);
}

py::array_t<double> Heston::initial_state(int n_sims) const {
    py::array_t<double> state({n_sims, 2});
    auto s = state.mutable_unchecked<2>();
    for (int i = 0; i < n_sims; ++i) {
        s(i, 0) = params_.spot;
        s(i, 1) = params_.v0;
    }
    return state;
}

} // namespace mc