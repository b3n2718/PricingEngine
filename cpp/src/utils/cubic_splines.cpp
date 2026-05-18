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
        throw std::invalid_argument("knots must be 1D");

if (coeffs_.shape(1) != 4)
    throw std::invalid_argument(
        "first coeff dimension must be 4"
    );

if (coeffs_.shape(0) != knots_.shape(0)-1)
    throw std::invalid_argument(
        "wrong number of spline segments"
    );
}

double CubicSpline::eval(double x) const {

    auto k = knots_.unchecked<1>();
    auto c = coeffs_.unchecked<2>();

    int i = find_segment(x);

    double dx = x - k(i);

    return c(i,0)
         + dx * (c(i,1)
         + dx * (c(i,2)
         + dx *  c(i,3)));
}

double CubicSpline::integral(double x, double y) const {

    double dx = (y - x) / 100.0;
    double sum = 0.0;

    for(int i = 0; i < 100; ++i) {
        sum += dx * eval(x + dx * (i + 0.5));
    }

    return sum;
}

int CubicSpline::find_segment(double x) const {

    auto k = knots_.unchecked<1>();

    if (x <= k(0))
        return 0;

    if (x >= k(k.shape(0)-1))
        return static_cast<int>(k.shape(0)) - 2;

    const double* begin = &k(0);
    const double* end   = begin + k.shape(0);

    auto it = std::upper_bound(begin, end, x);

    return static_cast<int>(std::distance(begin, it)) - 1;
}

}