#pragma once
#include "process_base.hpp"

namespace mc {

struct GammaVarianceParams {
    double spot;
    double vol;
    double risk_free_rate;
    double div_yield;
    double theta;
    double nu;
};

class GammaVariance : public ProcessBase {
public:
    explicit GammaVariance(const GammaVarianceParams& params);

    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    int state_dim() const override { return 1; }
    int noise_dim() const override { return 2; }

    // Initialen Zustand aufbauen — spot für alle n_sims
    py::array_t<double> initial_state(int n_sims) const;

private:
    GammaVarianceParams params_;
};

} // namespace mc