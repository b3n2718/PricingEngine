//hjm.cpp
#include "utils/cubic_splines.hpp"
#include "processes/hjm.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>




namespace mc {

namespace py = pybind11;

HJM::HJM(const HJMParams& params) : params_(params) {    
    build_splines();
    compute_alpha();
    type = "HJM";
}

py::array_t<double> HJM::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    auto s  = state.unchecked<2>();
    auto zr = z.unchecked<2>();        // GEÄNDERT: 2D statt 1D — [n_sims, noise_dim]
    int  n  = static_cast<int>(state.shape(0));
    auto tenors  = params_.tenors.unchecked<1>();

    py::array_t<double> result({n, static_cast<int>(params_.tenors.size())});
    auto r = result.mutable_unchecked<2>();    
    double sqrt_dt = std::sqrt(dt);
    
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < static_cast<int>(params_.tenors.size()); ++j)
        {
            r(i,j) = s(i,j) + alpha_[j] * dt;
            for (int k = 0; k < params_.n_factors; ++k){
                r(i,j) += vol(k, tenors(j)) * std::sqrt(dt) * zr(i,k);
            }
        }
    }
    return result;
}


py::array_t<double> HJM::initial_state(int n_sims) const {
    py::array_t<double> state({n_sims, static_cast<int>(params_.tenors.size())});
    auto s = state.mutable_unchecked<2>();
    auto r_forward = params_.r_forward.unchecked<1>();
    for (int i = 0; i < n_sims; ++i){
        for (int j = 0; j < static_cast<int>(params_.tenors.size()); ++j)
        s(i,j) = r_forward(j);
        }
    return state;
}

void HJM::compute_alpha() {
    // Alle Tenors auf einmal — vermeidet doppelte Schleife
    // Nutzt aus dass ∫_t^{T_j} σ ds eine kumulative Summe ist
    int    n_tenors = static_cast<int>(params_.tenors.size());
    auto tenors  = params_.tenors.unchecked<1>();

    alpha_.clear();
    alpha_.reserve(static_cast<int>(params_.tenors.size()));
    double temp = 0;
    for(int i = 0;i<n_tenors;i++){
        temp = 0;
        for(int j = 0; j < params_.n_factors; j++){
            temp += vol(j, tenors(i)) * splines_[j].integral(0, tenors[i]);
        }
        alpha_.push_back(temp);
    }
}


double HJM::vol(int k, double tau) const{
    auto std_scores = params_.std_scores.unchecked<1>();
    return splines_[k].eval(tau) * std_scores(k);
}

void HJM::build_splines() {

    splines_.clear();
    splines_.reserve(params_.n_factors);

    for (int k = 0; k < params_.n_factors; ++k) {

        splines_.emplace_back(
            params_.tenors,
            params_.spline_parameters[
                py::make_tuple(k)
            ].cast<py::array_t<double>>()
        );
    }
}

} // namespace mc