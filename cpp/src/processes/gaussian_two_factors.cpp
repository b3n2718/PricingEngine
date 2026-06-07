#include "processes/gaussian_two_factors.hpp"
#include <cmath>

namespace mc {

namespace py = pybind11;

G2pp::G2pp(const G2ppParams& params) : params_(params) {
    type = "G2PP";
}

py::array_t<double> G2pp::evolve(
    const py::array_t<double>& state,
    const py::array_t<double>& z,
    double dt) const
{
    // state: [n_sims, 2] — column 0: x(t), column 1: y(t)
    // z:     [n_sims, 2] — two independent standard-normal draws
    auto s  = state.unchecked<2>();
    auto zr = z.unchecked<2>();
    int  n  = static_cast<int>(state.shape(0));

    py::array_t<double> result({n, 2});
    auto r = result.mutable_unchecked<2>();

    double sqrt_dt = std::sqrt(dt);

    // Apply the Cholesky decomposition for the correlation ρ between x and y:
    //   dW_x = z₁ · √dt
    //   dW_y = (ρ·z₁ + √(1-ρ²)·z₂) · √dt
    //
    // This ensures Cov(dW_x, dW_y) = ρ dt while both z columns are
    // independent standard normals passed from the Python layer.
    double rho      = params_.rho;
    double sqrt_1r2 = std::sqrt(1.0 - rho * rho);

    for (int i = 0; i < n; ++i) {
        double x = s(i, 0);
        double y = s(i, 1);

        double dW_x = zr(i, 0) * sqrt_dt;
        double dW_y = (rho * zr(i, 0) + sqrt_1r2 * zr(i, 1)) * sqrt_dt;

        // Euler-Maruyama step for the two Ornstein-Uhlenbeck factors:
        //   x(t+dt) = x(t) - a·x(t)·dt + σ·dW_x
        //   y(t+dt) = y(t) - b·y(t)·dt + η·dW_y
        r(i, 0) = x - params_.a * x * dt + params_.sigma * dW_x;
        r(i, 1) = y - params_.b * y * dt + params_.eta   * dW_y;
    }

    return result;
}

py::array_t<double> G2pp::initial_state(int n_sims) const {
    // Both factors start at their initial values (typically 0).
    py::array_t<double> state({n_sims, 2});
    auto s = state.mutable_unchecked<2>();
    for (int i = 0; i < n_sims; ++i) {
        s(i, 0) = params_.x0;
        s(i, 1) = params_.y0;
    }
    return state;
}

} // namespace mc
