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
    auto s  = state.unchecked<2>();   // [n_sims, 2]
    auto zr = z.unchecked<2>();       // [n_sims, 2]
    int  n  = static_cast<int>(state.shape(0));

    py::array_t<double> result({n, 2});
    auto r = result.mutable_unchecked<2>();

    double sqrt_dt = std::sqrt(dt);

    // Cholesky für Korrelation zwischen x und y
    // z(:,0) → dW1, z(:,1) → dW2 unabhängig
    // dW_x = z1
    // dW_y = rho*z1 + sqrt(1-rho^2)*z2
    double rho     = params_.rho;
    double sqrt_1r = std::sqrt(1.0 - rho * rho);

    for (int i = 0; i < n; ++i) {
        double x = s(i, 0);
        double y = s(i, 1);

        double dW_x = zr(i, 0) * sqrt_dt;
        double dW_y = (rho * zr(i, 0) + sqrt_1r * zr(i, 1)) * sqrt_dt;

        // Euler-Maruyama:
        // dx = -a*x*dt + sigma*dW_x
        // dy = -b*y*dt + eta*dW_y
        r(i, 0) = x - params_.a * x * dt + params_.sigma * dW_x;
        r(i, 1) = y - params_.b * y * dt + params_.eta   * dW_y;
    }

    return result;
}

py::array_t<double> G2pp::initial_state(int n_sims) const {
    py::array_t<double> state({n_sims, 2});
    auto s = state.mutable_unchecked<2>();
    for (int i = 0; i < n_sims; ++i) {
        s(i, 0) = params_.x0;
        s(i, 1) = params_.y0;
    }
    return state;
}

} // namespace mc