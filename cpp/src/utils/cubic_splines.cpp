#include "utils/cubic_splines.hpp"
#include <stdexcept>
#include <algorithm>

namespace mc {
namespace py = pybind11;

CubicSpline::CubicSpline(const py::array_t<double>& knots,
                         const py::array_t<double>& coeffs)
    : knots_(knots), coeffs_(coeffs)
{
    if (knots_.ndim() != 1)
        throw std::invalid_argument("knots must be 1-D");

    // scipy CubicSpline stores coefficients as [n_segments, 4]
    if (coeffs_.shape(1) != 4)
        throw std::invalid_argument(
            "coeffs must have 4 columns (cubic polynomial coefficients)"
        );

    // Number of segments must equal number of intervals between knots
    if (coeffs_.shape(0) != knots_.shape(0) - 1)
        throw std::invalid_argument(
            "number of spline segments must equal len(knots) - 1"
        );
}

double CubicSpline::eval(double x) const {
    // Evaluate p(x) on the segment containing x:
    //   p(x) = c[i,0] + c[i,1]·dx + c[i,2]·dx² + c[i,3]·dx³
    // where dx = x - knot[i] and i is the segment index.
    auto k = knots_.unchecked<1>();
    auto c = coeffs_.unchecked<2>();

    int    i  = find_segment(x);
    double dx = x - k(i);

    // Horner's method for numerical stability and efficiency
    return c(i, 0)
         + dx * (c(i, 1)
         + dx * (c(i, 2)
         + dx *  c(i, 3)));
}

double CubicSpline::integral(double x, double y) const {
    // Numerical integration using the midpoint rule with 100 sub-intervals.
    // The HJM drift integral ∫_0^T σ_k(s) ds does not require high accuracy
    // since it is a smooth volatility function — 100 intervals is sufficient.
    double dx  = (y - x) / 100.0;
    double sum = 0.0;

    for (int i = 0; i < 100; ++i) {
        sum += dx * eval(x + dx * (i + 0.5));  // midpoint of sub-interval i
    }

    return sum;
}

int CubicSpline::find_segment(double x) const {
    // Binary search for the index i such that knots_[i] <= x < knots_[i+1].
    // Clamp to the first/last valid segment for out-of-range values so that
    // extrapolation uses the boundary polynomial (constant extrapolation of
    // the endpoint segment).
    auto k = knots_.unchecked<1>();

    if (x <= k(0))
        return 0;

    if (x >= k(k.shape(0) - 1))
        return static_cast<int>(k.shape(0)) - 2;  // last valid segment index

    const double* begin = &k(0);
    const double* end   = begin + k.shape(0);

    // upper_bound returns an iterator to the first knot > x, so the segment
    // index is one position to the left.
    auto it = std::upper_bound(begin, end, x);
    return static_cast<int>(std::distance(begin, it)) - 1;
}

} // namespace mc
