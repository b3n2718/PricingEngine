#pragma once
#include "process_base.hpp"

namespace mc {

struct HestonParams {
    double spot;
    double v0;      // Anfangsvarianz
    double kappa;   // Mean-Reversion Speed
    double theta;   // Long-run Varianz
    double xi;      // Vol of Vol
    double rho;     // Korrelation Spot/Varianz
    double risk_free_rate;
    double div_yield;
};

class Heston : public ProcessBase {
public:
    explicit Heston(const HestonParams& params);

    // state: [n_sims, 2] — Spalte 0: Spot, Spalte 1: Varianz
    // z:     [n_sims, 2] — zwei unabhängige Brownschen Bewegungen
    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    int state_dim() const override { return 2; }
    int noise_dim() const override { return 2; }

    py::array_t<double> initial_state(int n_sims) const;

private:
    HestonParams params_;

    // Reflection scheme für negative Varianz
    double apply_reflection(double v) const;
};

} // namespace mc