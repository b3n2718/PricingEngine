#include "processes/gbm.hpp"
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace mc {

GBM::GBM(const GBMParams& params) : params_(params) {
    if (params_.vol < 0)
        throw std::invalid_argument("Volatility must be non-negative");
    if (params_.spot <= 0)
        throw std::invalid_argument("Spot must be positive");
    type = "GBM";
}

py::array_t<double> GBM::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    auto s  = state.unchecked<1>();
    auto zr = z.unchecked<2>();        // GEÄNDERT: 2D statt 1D — [n_sims, noise_dim]
    int  n  = static_cast<int>(state.shape(0));

    py::array_t<double> result(n);
    auto r = result.mutable_unchecked<1>();

    double drift     = (params_.risk_free_rate
                       - params_.div_yield
                       - 0.5 * params_.vol * params_.vol) * dt;
    double diffusion = params_.vol * std::sqrt(dt);

    for (int i = 0; i < n; ++i){
        r(i) = s(i) * std::exp(drift + diffusion * zr(i, 0));  // GEÄNDERT: zr(i,0)
    }
    return result;
}

py::array_t<double> GBM::initial_state(int n_sims) const {
    py::array_t<double> state(n_sims);
    auto s = state.mutable_unchecked<1>();
    for (int i = 0; i < n_sims; ++i)
        s(i) = params_.spot;
    return state;
}

} // namespace mc