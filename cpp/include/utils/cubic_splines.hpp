// cubic_spline.hpp — nur Deklaration
#pragma once
#include <vector>
#include <pybind11/numpy.h>

namespace mc {
namespace py = pybind11;

class CubicSpline {
public:
    CubicSpline(const py::array_t<double>& knots,
                const py::array_t<double>& coeffs);

    double eval(double x) const;
    double integral(double x, double y) const;

private:
    const py::array_t<double>         knots_;
    const py::array_t<double>         coeffs_;

    int find_segment(double x) const;
};

}