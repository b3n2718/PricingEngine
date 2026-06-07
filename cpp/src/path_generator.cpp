// path_generator.cpp
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include "path_generator.hpp"
#include "processes/gbm.hpp"
#include "processes/heston.hpp"
#include "processes/gamma_variance.hpp"
#include "processes/vasicek.hpp"
#include "processes/cir.hpp"
#include "processes/hjm.hpp"
#include "processes/gaussian_two_factors.hpp"
#include <stdexcept>

namespace mc {

namespace py = pybind11;

py::dict PathGenerator::generate(
    const std::vector<py::dict>& params_list,
    const std::vector<std::string>& asset_ids,
    const py::array_t<double>& z,
    double dt)
{
    auto n_sims   = static_cast<int>(z.shape(0));
    auto n_steps  = static_cast<int>(z.shape(1));
    auto n_assets = static_cast<int>(params_list.size());

    py::dict result;
    int noise_offset = 0;  // tracks where in the noise axis the current asset starts

    for (int a = 0; a < n_assets; ++a) {
        auto process = build_process(params_list[a]);
        int  ndim    = process->noise_dim();

        std::string path_type = params_list[a]["path_type"].cast<std::string>();

        // Manually copy the noise slice for this asset from the global tensor.
        // The global z has shape [n_sims, n_steps, total_noise_dim]; each asset
        // owns a contiguous block of ndim columns starting at noise_offset.
        py::array_t<double> z_asset({n_sims, n_steps, ndim});
        auto za = z_asset.mutable_unchecked<3>();
        auto zr = z.unchecked<3>();

        for (int i = 0; i < n_sims; ++i)
            for (int t = 0; t < n_steps; ++t)
                for (int d = 0; d < ndim; ++d)
                    za(i, t, d) = zr(i, t, noise_offset + d);

        // Dispatch to the correct simulation loop based on the path type.
        py::array_t<double> path;
        if (path_type == "spot") {
            path = simulate_process_spot(*process, z_asset, dt, n_sims, n_steps);
        } else if (path_type == "forward") {
            int n_tenors = params_list[a]["tenors"].cast<py::array_t<double>>().shape(0);
            path = simulate_process_forward(*process, z_asset, dt, n_sims, n_steps, n_tenors);
        } else {
            throw std::invalid_argument("Unknown path type: " + path_type);
        }

        result[asset_ids[a].c_str()] = path;
        noise_offset += ndim;  // advance past this asset's noise columns
    }

    return result;
}

std::unique_ptr<ProcessBase> PathGenerator::build_process(
    const py::dict& params)
{
    // Construct the concrete process from the parameter dict sent by Python.
    // The "type" key selects the class; remaining keys are model parameters.
    std::string type = params["type"].cast<std::string>();

    if (type == "GBM") {
        return std::make_unique<GBM>(GBMParams{
            params["spot"].cast<double>(),
            params["vol"].cast<double>(),
            params["risk_free_rate"].cast<double>(),
            params["div_yield"].cast<double>()
        });
    }
    if (type == "HESTON") {
        return std::make_unique<Heston>(HestonParams{
            params["spot"].cast<double>(),
            params["v0"].cast<double>(),
            params["kappa"].cast<double>(),
            params["theta"].cast<double>(),
            params["xi"].cast<double>(),
            params["rho"].cast<double>(),
            params["risk_free_rate"].cast<double>(),
            params["div_yield"].cast<double>()
        });
    }
    if (type == "VARIANCEGAMMA") {
        return std::make_unique<VarianceGamma>(VarianceGammaParams{
            params["spot"].cast<double>(),
            params["vol"].cast<double>(),
            params["risk_free_rate"].cast<double>(),
            params["div_yield"].cast<double>(),
            params["theta"].cast<double>(),
            params["nu"].cast<double>()
        });
    }
    if (type == "VASICEK") {
        return std::make_unique<Vasicek>(VasicekParams{
            params["r_spot"].cast<double>(),
            params["kappa"].cast<double>(),
            params["theta"].cast<double>(),
            params["vol"].cast<double>()
        });
    }
    if (type == "CIR") {
        return std::make_unique<CIR>(CIRParams{
            params["r_spot"].cast<double>(),
            params["kappa"].cast<double>(),
            params["theta"].cast<double>(),
            params["vol"].cast<double>()
        });
    }
    if (type == "HJM") {
        return std::make_unique<HJM>(HJMParams{
            params["r_forward"].cast<py::array_t<double>>(),
            params["std_scores"].cast<py::array_t<double>>(),
            params["spline_parameters"].cast<py::array_t<double>>(),
            params["tenors"].cast<py::array_t<double>>(),
            params["num_vol_comp"].cast<int>()
        });
    }
    if (type == "G2PP") {
        return std::make_unique<G2pp>(G2ppParams{
            params["a"].cast<double>(),
            params["b"].cast<double>(),
            params["sigma"].cast<double>(),
            params["eta"].cast<double>(),
            params["rho"].cast<double>(),
            params["x0"].cast<double>(),
            params["y0"].cast<double>()
        });
    }

    throw std::invalid_argument("Unknown process type: " + type);
}

py::array_t<double> PathGenerator::simulate_process_spot(
    const ProcessBase& process,
    const py::array_t<double>& z_asset,
    double dt,
    int n_sims,
    int n_steps)
{
    // Allocate the output array.  Shape depends on the process:
    //   - state_dim == 1 (GBM, Vasicek, CIR, VG): [n_sims, n_steps]
    //   - G2PP (state_dim == 2, stores both factors): [n_sims, 2, n_steps]
    //   - Heston (state_dim == 2, spot only recorded): [n_sims, n_steps]
    py::array_t<double> path;

    if (process.state_dim() == 1) {
        path = py::array_t<double>({n_sims, n_steps});
    } else if (process.type == "G2PP") {
        path = py::array_t<double>({n_sims, 2, n_steps});  // store both factors
    } else {
        path = py::array_t<double>({n_sims, n_steps});     // Heston: spot only
    }

    // Seed the state with the process-specific initial values.
    py::array_t<double> state = process.initial_state(n_sims);

    auto zr   = z_asset.unchecked<3>();  // [n_sims, n_steps, noise_dim]
    int  ndim = process.noise_dim();

    for (int t = 0; t < n_steps; ++t) {
        // Extract the noise slice for this time step: [n_sims, noise_dim]
        py::array_t<double> z_t({n_sims, ndim});
        auto zt = z_t.mutable_unchecked<2>();
        for (int i = 0; i < n_sims; ++i)
            for (int d = 0; d < ndim; ++d)
                zt(i, d) = zr(i, t, d);

        state = process.evolve(state, z_t, dt);

        // Write the relevant component(s) of the new state into the path array.
        if (process.state_dim() == 1) {
            // Scalar state: write the single value directly.
            auto p = path.mutable_unchecked<2>();
            auto s = state.unchecked<1>();
            for (int i = 0; i < n_sims; ++i)
                p(i, t) = s(i);
        } else if (process.type == "HESTON") {
            // Heston: state is [n_sims, 2]; record only column 0 (spot).
            auto p = path.mutable_unchecked<2>();
            auto s = state.unchecked<2>();
            for (int i = 0; i < n_sims; ++i)
                p(i, t) = s(i, 0);
        } else if (process.type == "G2PP") {
            // G2++: record both factors x (col 0) and y (col 1).
            auto p = path.mutable_unchecked<3>();
            auto s = state.unchecked<2>();
            for (int i = 0; i < n_sims; ++i) {
                p(i, 0, t) = s(i, 0);  // x(t)
                p(i, 1, t) = s(i, 1);  // y(t)
            }
        }
    }

    return path;
}

py::array_t<double> PathGenerator::simulate_process_forward(
    const ProcessBase& process,
    const py::array_t<double>& z_asset,
    double dt,
    int n_sims,
    int n_steps,
    int n_tenors)
{
    // HJM output: full forward curve at every time step.
    // Shape: [n_sims, n_tenors, n_steps]
    py::array_t<double> path({n_sims, n_tenors, n_steps});
    auto p = path.mutable_unchecked<3>();

    py::array_t<double> state = process.initial_state(n_sims);  // [n_sims, n_tenors]

    auto zr   = z_asset.unchecked<3>();  // [n_sims, n_steps, n_factors]
    int  ndim = process.noise_dim();

    for (int t = 0; t < n_steps; ++t) {
        // Extract noise slice for this step: [n_sims, n_factors]
        py::array_t<double> z_t({n_sims, ndim});
        auto zt = z_t.mutable_unchecked<2>();
        for (int i = 0; i < n_sims; ++i)
            for (int d = 0; d < ndim; ++d)
                zt(i, d) = zr(i, t, d);

        state = process.evolve(state, z_t, dt);  // returns [n_sims, n_tenors]

        // Copy the entire forward curve into the output array at time t.
        auto s = state.unchecked<2>();
        for (int i = 0; i < n_sims; ++i)
            for (int j = 0; j < n_tenors; ++j)
                p(i, j, t) = s(i, j);
    }

    return path;
}

} // namespace mc
