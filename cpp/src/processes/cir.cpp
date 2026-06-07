#include "processes/cir.hpp"
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>

namespace mc {

namespace py = pybind11;

CIR::CIR(const CIRParams& params) : params_(params) {
    if (params_.vol < 0)
        throw std::invalid_argument("Volatility must be non-negative");
    type = "CIR";
}

py::array_t<double> CIR::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    auto s  = state.unchecked<1>();  // current short rates, shape [n_sims]
    auto zr = z.unchecked<2>();      // noise matrix [n_sims, 1]
    int  n  = static_cast<int>(state.shape(0));

    py::array_t<double> result(n);
    auto r = result.mutable_unchecked<1>();

    double sqrt_dt = std::sqrt(dt);

    // Euler-Maruyama step:
    //   r(t+dt) = r(t) + κ(θ - r(t)) dt + σ √(max(r(t),0)) √dt · Z
    //
    // The max(r(t), 0) guard handles rare cases where Euler produces a
    // slightly negative rate; the Feller condition ensures this is uncommon.
    for (int i = 0; i < n; ++i) {
        r(i) = s(i) + params_.kappa * (params_.theta - s(i)) * dt
                    + std::sqrt(std::max(s(i), 0.0)) * params_.vol * sqrt_dt * zr(i, 0);
    }
    return result;
}

py::array_t<double> CIR::initial_state(int n_sims) const {
    // All paths start at the current market short rate.
    py::array_t<double> state(n_sims);
    auto s = state.mutable_unchecked<1>();
    for (int i = 0; i < n_sims; ++i)
        s(i) = params_.r_spot;
    return state;
}

} // namespace mc
