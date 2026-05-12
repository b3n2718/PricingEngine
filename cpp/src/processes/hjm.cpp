#include "processes/hjm.hpp"
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>




namespace mc {
class SigmaComponent {
public:
    SigmaComponent(vector<double> knots,
                   vector<array<double, 4>> coeffs,double score_std)
        : knots_(move(knots)), coeffs_(move(coeffs)), score_std_(score_std)
    {
        if (knots_.size() != coeffs_.size() + 1)
            throw invalid_argument("Knots must be one longer than coeffs.");
    }

    // Wert von sigma_i(T)
    double operator()(double T) const {
        if (T <= knots_.front()) return evalSpline(0, T);
        if (T >= knots_.back()) return evalSpline(coeffs_.size() - 1, T);

        auto it = upper_bound(knots_.begin(), knots_.end(), T);
        size_t idx = distance(knots_.begin(), it) - 1;

        return evalSpline(idx, T);
    }

    // Integral von sigma_i von t bis T
    double integral(double t, double T) const {
        if (T < t) swap(t, T);
        t = max(t, knots_.front());
        T = min(T, knots_.back());

        if (t >= T) return 0.0;

        auto it_t = upper_bound(knots_.begin(), knots_.end(), t);
        auto it_T = upper_bound(knots_.begin(), knots_.end(), T);

        size_t idx_t = distance(knots_.begin(), it_t) - 1;
        size_t idx_T = distance(knots_.begin(), it_T) - 1;

        double result = 0.0;

        if (idx_t == idx_T) {
            return integralSpline(idx_t, t, T);
        }

        result += integralSpline(idx_t, t, knots_[idx_t + 1]);
        for (size_t i = idx_t + 1; i < idx_T; ++i)
            result += integralSpline(i, knots_[i], knots_[i + 1]);

        result += integralSpline(idx_T, knots_[idx_T], T);
        return result;
    }

private:
    vector<double> knots_;  // Knotenpunkte x_0, ..., x_n
    vector<array<double, 4>> coeffs_;  // [a, b, c, d] für jedes Intervall
    double score_std_;

    double evalSpline(size_t idx, double T) const {
        double dt = T - knots_[idx];
        const auto& c = coeffs_[idx];
        return score_std_ * (c[0] + c[1] * dt + c[2] * dt * dt + c[3] * dt * dt * dt);
    }

    double integralSpline(size_t idx, double t, double T) const {
        double x1 = t - knots_[idx];
        double x2 = T - knots_[idx];
        const auto& c = coeffs_[idx];

        auto F = [&](double x) {
            return c[0] * x +
                   c[1] * x * x / 2.0 +
                   c[2] * x * x * x / 3.0 +
                   c[3] * x * x * x * x / 4.0;
        };
        return score_std_ * (F(x2) - F(x1));
    }
};

namespace py = pybind11;

HJM::HJM(const HJMParams& params) : params_(params) {}

py::array_t<double> HJM::evolve(
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

py::array_t<double> HJM::initial_state(int n_sims) const {
    py::array_t<double> state(n_sims);
    auto s = state.mutable_unchecked<2>();
    for (int i = 0; i < n_sims; ++i){
        for (int j = =; j < params_.n_tenors; ++j)
        s(i,j) = params_.r_forward[j];
        }
    return state;
}

std::vector<double> HJM::alpha_vec(double t) const {
    // Alle Tenors auf einmal — vermeidet doppelte Schleife
    // Nutzt aus dass ∫_t^{T_j} σ ds eine kumulative Summe ist
    int    n_tenors = static_cast<int>(params_.tenors.size());
    double dT       = params_.tenors[1] - params_.tenors[0];

    std::vector<double> result(n_tenors, 0.0);

    for (int k = 0; k < params_.n_factors; ++k) {
        // Sigma-Vektor für diesen Faktor
        std::vector<double> sigma_k(n_tenors);
        for (int j = 0; j < n_tenors; ++j)
            sigma_k[j] = vol(k, t, params_.tenors[j]);

        // Kumulative Summe = ∫_t^{T_j} σ_k ds
        std::vector<double> integral(n_tenors, 0.0);
        double cumsum = 0.0;
        for (int j = 0; j < n_tenors; ++j) {
            if (params_.tenors[j] >= t) {
                cumsum     += sigma_k[j] * dT;
                integral[j] = cumsum;
            }
        }

        // α(t, T_j) += σ_k(t, T_j) * ∫_t^{T_j} σ_k ds
        for (int j = 0; j < n_tenors; ++j)
            result[j] += sigma_k[j] * integral[j];
    }
    return result;
}

} // namespace mc