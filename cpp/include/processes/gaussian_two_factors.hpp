#pragma once
#include "process_base.hpp"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace mc {

namespace py = pybind11;

struct G2ppParams {
    double a;           // Mean-Reversion Speed x
    double b;           // Mean-Reversion Speed y
    double sigma;       // Volatilität x
    double eta;         // Volatilität y
    double rho;         // Korrelation x/y
    double x0;          // Anfangswert x — typisch 0
    double y0;          // Anfangswert y — typisch 0
};

class G2pp : public ProcessBase {
public:
    explicit G2pp(const G2ppParams& params);

    // state: [n_sims, 2] — Spalte 0: x, Spalte 1: y
    // z:     [n_sims, 2] — zwei unabhängige Brownschen Bewegungen
    py::array_t<double> evolve(
        const py::array_t<double>& state,
        const py::array_t<double>& z,
        double dt
    ) const override;

    py::array_t<double> initial_state(int n_sims) const override;

    int state_dim() const override { return 2; }
    int noise_dim() const override { return 2; }

private:
    G2ppParams params_;

};

} // namespace mc