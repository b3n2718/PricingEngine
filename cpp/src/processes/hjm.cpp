// hjm.cpp
#include "utils/cubic_splines.hpp"
#include "processes/hjm.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>

namespace mc {

namespace py = pybind11;

HJM::HJM(const HJMParams& params) : params_(params) {
    build_splines();   // fit one CubicSpline per PCA component
    compute_alpha();   // pre-compute the no-arbitrage drift vector
    type = "HJM";
}

py::array_t<double> HJM::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    // state: [n_sims, n_tenors] — full forward curve for each path
    // z:     [n_sims, n_factors] — one standard-normal draw per PCA component
    auto s      = state.unchecked<2>();
    auto zr     = z.unchecked<2>();
    int  n      = static_cast<int>(state.shape(0));
    auto tenors = params_.tenors.unchecked<1>();

    py::array_t<double> result({n, static_cast<int>(params_.tenors.size())});
    auto r = result.mutable_unchecked<2>();

    double sqrt_dt = std::sqrt(dt);

    // HJM Euler step for each path i and tenor T_j:
    //   f(t+dt, T_j) = f(t, T_j) + α_j·dt + Σ_k σ_k(T_j)·√dt·Z_k
    //
    // α_j is the pre-computed no-arbitrage drift (see compute_alpha).
    // vol(k, τ) = spline_k(τ) · std_scores[k] recovers historical vol units.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < static_cast<int>(params_.tenors.size()); ++j) {
            r(i, j) = s(i, j) + alpha_[j] * dt;
            for (int k = 0; k < params_.n_factors; ++k) {
                r(i, j) += vol(k, tenors(j)) * sqrt_dt * zr(i, k);
            }
        }
    }
    return result;
}

py::array_t<double> HJM::initial_state(int n_sims) const {
    // All paths start with the market forward curve (same for every path).
    py::array_t<double> state({n_sims, static_cast<int>(params_.tenors.size())});
    auto s         = state.mutable_unchecked<2>();
    auto r_forward = params_.r_forward.unchecked<1>();
    for (int i = 0; i < n_sims; ++i) {
        for (int j = 0; j < static_cast<int>(params_.tenors.size()); ++j)
            s(i, j) = r_forward(j);
    }
    return state;
}

void HJM::compute_alpha() {
    // HJM no-arbitrage condition:
    //   α(t, T) = Σ_k σ_k(T) · ∫_0^T σ_k(s) ds
    //
    // This drift is deterministic and depends only on T (not on t or the state),
    // so it can be computed once and stored in alpha_.
    int  n_tenors = static_cast<int>(params_.tenors.size());
    auto tenors   = params_.tenors.unchecked<1>();

    alpha_.clear();
    alpha_.reserve(n_tenors);

    for (int i = 0; i < n_tenors; ++i) {
        double alpha_j = 0.0;
        for (int k = 0; k < params_.n_factors; ++k) {
            // σ_k(T) · ∫_0^T σ_k(s) ds
            alpha_j += vol(k, tenors(i)) * splines_[k].integral(0.0, tenors(i));
        }
        alpha_.push_back(alpha_j);
    }
}

double HJM::vol(int k, double tau) const {
    // Evaluate the k-th volatility component at tenor τ.
    // The PCA eigenvector (normalised) is scaled back to historical vol units
    // by multiplying with the standard deviation of the k-th score series.
    auto std_scores = params_.std_scores.unchecked<1>();
    return splines_[k].eval(tau) * std_scores(k);
}

void HJM::build_splines() {
    // Construct one CubicSpline per PCA component from the spline coefficient
    // array produced by scipy's CubicSpline in Python.
    splines_.clear();
    splines_.reserve(params_.n_factors);

    for (int k = 0; k < params_.n_factors; ++k) {
        // Extract the coefficient matrix for component k: shape [n_segments, 4].
        splines_.emplace_back(
            params_.tenors,
            params_.spline_parameters[py::make_tuple(k)].cast<py::array_t<double>>()
        );
    }
}

} // namespace mc
