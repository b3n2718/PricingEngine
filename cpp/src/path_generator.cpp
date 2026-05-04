// src/path_generator.cpp — komplett überarbeitet

#include "path_generator.hpp"
#include "processes/gbm.hpp"
#include "processes/heston.hpp"
//#include "process/gamma_variance.hpp"
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

        // z-Slice für dieses Asset manuell kopieren: [n_sims, n_steps, ndim]
        // Kein Python-Slicing — direkt in einen neuen Buffer kopieren
        py::array_t<double> z_asset({n_sims, n_steps, ndim});
        auto za  = z_asset.mutable_unchecked<3>();
        auto zr  = z.unchecked<3>();

        for (int i = 0; i < n_sims; ++i)
            for (int t = 0; t < n_steps; ++t)
                for (int d = 0; d < ndim; ++d)
                    za(i, t, d) = zr(i, t, noise_offset + d);

        auto spot_path = simulate_process(
            *process, z_asset, dt, n_sims, n_steps
        );

        result[asset_ids[a].c_str()] = spot_path;
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
 /*   if (type == "GAMMAVARIANCE") {
        return std::make_unique<GammaVariance>(GammaVarianceParams{
            params["spot"].cast<double>(),
            params["risk_free_rate"].cast<double>(),
            params["theta"].cast<double>(),
            params["nu"].cast<double>()
        });
    }*/
    throw std::invalid_argument("Unbekannter Prozesstyp: " + type);
}


py::array_t<double> PathGenerator::simulate_process(
    const ProcessBase& process,
    const py::array_t<double>& z_asset,
    double dt,
    int n_sims,
    int n_steps)
{
    // Ergebnis-Array: nur Spot-Pfad [n_sims, n_steps]
    py::array_t<double> path({n_sims, n_steps});
    auto p = path.mutable_unchecked<2>();

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

        // Spot aus Zustand lesen
        if (process.state_dim() == 1) {
            // GBM: state ist [n_sims]
            auto s = state.unchecked<1>();
            for (int i = 0; i < n_sims; ++i)
                p(i, t) = s(i);
        } else {
            // Heston: state ist [n_sims, 2] — Spalte 0 ist Spot
            auto s = state.unchecked<2>();
            for (int i = 0; i < n_sims; ++i)
                p(i, t) = s(i, 0);
        }
    }

    return path;
}

} // namespace mc