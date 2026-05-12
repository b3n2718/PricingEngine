#pragma once
#include "process_base.hpp"

namespace mc {

struct VasicekParams {
    double r_spot;
    double kappa;
    double theta;
    double vol;
};

class Vasicek : public ProcessBase {
public:
    explicit Vasicek(const VasicekParams& params);

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
    VasicekParams params_;
};

} // namespace mc