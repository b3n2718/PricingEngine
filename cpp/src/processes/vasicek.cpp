#include "processes/vasicek.hpp"
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>




namespace mc {

namespace py = pybind11;

Vasicek::Vasicek(const VasicekParams& params) : params_(params) {
    if (params_.vol < 0)
        throw std::invalid_argument("Volatility must be non-negative");
}

py::array_t<double> Vasicek::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    auto s  = state.unchecked<1>();
    auto zr = z.unchecked<2>();        // GEÄNDERT: 2D statt 1D — [n_sims, noise_dim]
    int  n  = static_cast<int>(state.shape(0));

    py::array_t<double> result(n);
    auto r = result.mutable_unchecked<1>();    
    double sqrt_dt = std::sqrt(dt);
    
    for (int i = 0; i < n; ++i){
        r(i) = s(i) + params_.kappa * (params_.theta - s(i)) * dt + params_.vol * sqrt_dt * zr(i,0);  // GEÄNDERT: zr(i,0)
    }
    return result;
}

py::array_t<double> Vasicek::initial_state(int n_sims) const {
    py::array_t<double> state(n_sims);
    auto s = state.mutable_unchecked<1>();
    for (int i = 0; i < n_sims; ++i)
        s(i) = params_.r_spot;
    return state;
}

} // namespace mc