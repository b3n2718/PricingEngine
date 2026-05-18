// src/path_generator.cpp
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
    auto n_sims    = static_cast<int>(z.shape(0));
    auto n_steps   = static_cast<int>(z.shape(1));
    auto n_assets  = static_cast<int>(params_list.size());

    py::dict result;
    int noise_offset = 0;

    for (int a = 0; a < n_assets; ++a) {
        auto process = build_process(params_list[a]);
        int  ndim    = process->noise_dim();

        std::string path_type = params_list[a]["path_type"].cast<std::string>();

        // z-Slice für dieses Asset manuell kopieren: [n_sims, n_steps, ndim]
        // Kein Python-Slicing — direkt in einen neuen Buffer kopieren
        py::array_t<double> z_asset({n_sims, n_steps, ndim});
        auto za  = z_asset.mutable_unchecked<3>();
        auto zr  = z.unchecked<3>();

        for (int i = 0; i < n_sims; ++i)
            for (int t = 0; t < n_steps; ++t)
                for (int d = 0; d < ndim; ++d)
                    za(i, t, d) = zr(i, t, noise_offset + d);

        py::array_t<double> path;
        if (path_type == "spot"){
            path = simulate_process_spot(
                *process, z_asset, dt, n_sims, n_steps
            );
        } else if (path_type == "forward") 
        {
            int n_tenors = params_list[a]["tenors"].cast<py::array_t<double>>().shape(0);
            path = simulate_process_forward(
                *process, z_asset, dt, n_sims, n_steps, n_tenors
            );
        } else
        {
            throw std::invalid_argument("Unbekannter Pfadtyp: " + path_type);
        }

        result[asset_ids[a].c_str()] = path;
        noise_offset += ndim;
    }

    return result;
}

std::unique_ptr<ProcessBase> PathGenerator::build_process(
    const py::dict& params)
{
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
        params["std_scores"].cast<py::array_t<double>>(),                              // pos 2
        params["spline_parameters"].cast<py::array_t<double>>(), // pos 3
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
    throw std::invalid_argument("Unbekannter Prozesstyp: " + type);
}


py::array_t<double> PathGenerator::simulate_process_spot(
    const ProcessBase& process,
    const py::array_t<double>& z_asset,
    double dt,
    int n_sims,
    int n_steps)
{
    // Ergebnis-Array: nur Spot-Pfad [n_sims, n_steps]
    py::array_t<double> path;

    if (process.state_dim() == 1) {
        path = py::array_t<double>({n_sims, n_steps});
    }
    else if (process.type == "G2PP") {
        path = py::array_t<double>({n_sims, 2, n_steps});
    }
    else {
        path = py::array_t<double>({n_sims, n_steps});
    }
    
    // Anfangszustand — jetzt über Basisklasse aufrufbar
    py::array_t<double> state = process.initial_state(n_sims);

    auto zr = z_asset.unchecked<3>();   // [n_sims, n_steps, noise_dim]
    int  ndim = process.noise_dim();
    

    for (int t = 0; t < n_steps; ++t) {

        // z für diesen Zeitschritt: [n_sims, noise_dim]
        py::array_t<double> z_t({n_sims, ndim});
        auto zt = z_t.mutable_unchecked<2>();
        for (int i = 0; i < n_sims; ++i)
            for (int d = 0; d < ndim; ++d)
                zt(i, d) = zr(i, t, d);

        state = process.evolve(state, z_t, dt);
        //py::print(process.type);

        // Spot aus Zustand lesen
        if (process.state_dim() == 1) {
            // GBM: state ist [n_sims]
            auto p = path.mutable_unchecked<2>();
            auto s = state.unchecked<1>();
            for (int i = 0; i < n_sims; ++i)
                p(i, t) = s(i);
        } else if(process.type=="HESTON"){
            // Heston: state ist [n_sims, 2] — Spalte 0 ist Spot
            auto p = path.mutable_unchecked<2>();
            auto s = state.unchecked<2>();
            for (int i = 0; i < n_sims; ++i)
                p(i, t) = s(i, 0);
        }else if(process.type=="G2PP"){
            // Heston: state ist [n_sims, 2] — Spalte 0 ist Spot
            auto p = path.mutable_unchecked<3>();
            auto s = state.unchecked<2>();
            for (int i = 0; i < n_sims; ++i){
                p(i, 0, t) = s(i, 0);
                p(i, 1, t) = s(i, 1);
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
    // Ergebnis-Array: nur Spot-Pfad [n_sims, n_steps]
    py::array_t<double> path({n_sims, n_tenors, n_steps});
    auto p = path.mutable_unchecked<3>();

    // Anfangszustand — jetzt über Basisklasse aufrufbar
    py::array_t<double> state = process.initial_state(n_sims);

    auto zr = z_asset.unchecked<3>();   // [n_sims, n_steps, noise_dim]
    int  ndim = process.noise_dim();

    for (int t = 0; t < n_steps; ++t) {

        // z für diesen Zeitschritt: [n_sims, noise_dim]
        py::array_t<double> z_t({n_sims, ndim});
        auto zt = z_t.mutable_unchecked<2>();
        for (int i = 0; i < n_sims; ++i)
            for (int d = 0; d < ndim; ++d)
                zt(i, d) = zr(i, t, d);

        state = process.evolve(state, z_t, dt);

        auto s = state.unchecked<2>();
        for (int i = 0; i < n_sims; ++i)
        {
            for (int j = 0; j < n_tenors; ++j)
            {
                p(i, j, t) = s(i,j);
            }
            
        }
    }

    return path;
}

} // namespace mc