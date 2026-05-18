//hjm.hpp
#pragma once
#include "process_base.hpp"
#include "utils/cubic_splines.hpp"

namespace mc {

struct HJMParams {
    const py::array_t<double> r_forward;         
    const py::array_t<double> std_scores;         
    const py::array_t<double> spline_parameters;  
    const py::array_t<double> tenors;             
    int                 n_factors;
};
class HJM : public ProcessBase {
public:
    explicit HJM(const HJMParams& params);

    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    int state_dim() const override { return static_cast<int>(params_.tenors.size()); }
    int noise_dim() const override { return params_.n_factors; }

    // Initialen Zustand aufbauen — spot für alle n_sims
    py::array_t<double> initial_state(int n_sims) const;

private:
    HJMParams params_;
    std::vector<CubicSpline> splines_;   // gebaut im Konstruktor
    std::vector<double> alpha_;

    void build_splines();
        // Volatilität σ_k(t, T)
    double vol(int k, double tau) const;

    // HJM Drift α(t, T_j) — Hilfsfunktion
    void compute_alpha();
};



} // namespace mc