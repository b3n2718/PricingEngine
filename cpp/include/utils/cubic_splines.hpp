// cubic_splines.hpp — declaration only
#pragma once
#include <vector>
#include <pybind11/numpy.h>

namespace mc {
namespace py = pybind11;

/**
 * @brief Piecewise cubic spline for evaluating HJM volatility components.
 *
 * Stores the knot positions and polynomial coefficients produced by
 * scipy.interpolate.CubicSpline (passed from Python as NumPy arrays).
 * The spline represents one PCA volatility component σ_k(τ) as a function
 * of the tenor τ.
 *
 * Coefficient layout
 * ------------------
 * For segment i (covering [knot_i, knot_{i+1}]):
 *
 *   p(x) = c(i,0) + c(i,1)·dx + c(i,2)·dx² + c(i,3)·dx³
 *
 * where dx = x - knot_i.  This matches scipy's coefficient ordering.
 *
 * Extrapolation behaviour
 * -----------------------
 * - Below the first knot: the first segment is used (constant extrapolation
 *   of the leftmost polynomial).
 * - Above the last knot:  the last segment is used.
 */
class CubicSpline {
public:
    /**
     * @brief Construct from knot positions and polynomial coefficients.
     *
     * @param knots   1-D array of knot positions (strictly increasing), shape [n_knots].
     * @param coeffs  2-D array of polynomial coefficients, shape [n_segments, 4]
     *                where n_segments = n_knots - 1.
     * @throws std::invalid_argument if shapes are inconsistent.
     */
    CubicSpline(const py::array_t<double>& knots,
                const py::array_t<double>& coeffs);

    /**
     * @brief Evaluate the spline at a single point x.
     *
     * @param x  Evaluation point (e.g. a tenor value in years).
     * @return   Spline value p(x).
     */
    double eval(double x) const;

    /**
     * @brief Numerically integrate the spline over [x, y] using the midpoint rule.
     *
     * Uses 100 equal sub-intervals.  Sufficient accuracy for the HJM drift
     * computation where the integral ∫σ_k(s)ds appears.
     *
     * @param x  Lower integration bound.
     * @param y  Upper integration bound.
     * @return   ∫_x^y p(s) ds (approximate).
     */
    double integral(double x, double y) const;

private:
    const py::array_t<double> knots_;   ///< Knot positions, shape [n_knots].
    const py::array_t<double> coeffs_;  ///< Polynomial coefficients, shape [n_segments, 4].

    /**
     * @brief Binary search for the segment index containing x.
     *
     * Returns the index i such that knots_[i] ≤ x < knots_[i+1], clamped
     * to [0, n_segments-1] for out-of-range values.
     */
    int find_segment(double x) const;
};

} // namespace mc
