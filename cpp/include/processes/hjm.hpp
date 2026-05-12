#pragma once
#include "process_base.hpp"

namespace mc {

struct HJMParams {
    std::vector<std::vector<double>> vol_params;   // [n_factors][4]
    std::vector<double>              initial_curve; // f(0, T_j) — [n_tenors]
    std::vector<double>              tenors;        // [n_tenors]
    int                              n_tenors;
    int                              n_factors;
};
class HJM : public ProcessBase {
public:
    explicit HJM(const HJMParams& params);

    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    int state_dim() const override { return 1; }
    int noise_dim() const override { return 1; }

    // Initialen Zustand aufbauen — spot für alle n_sims
    py::array_t<double> initial_state(int n_sims) const;

private:
    HJMParams params_;

        // Volatilität σ_k(t, T)
    double vol(int k, double t, double T) const;

    // HJM Drift α(t, T_j) — Hilfsfunktion
    double alpha(double t, double T_j) const;

    // Drift für alle Tenors auf einmal — effizienter
    std::vector<double> alpha_vec(double t) const;
};



} // namespace mc