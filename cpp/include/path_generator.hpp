#pragma once
#include "processes/process_base.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <memory>

namespace mc {

class PathGenerator {
public:

    // Hauptmethode — aufgerufen aus Python über pybind
    // params_list:  Liste von Dicts, eines pro Asset
    // asset_ids:    Namen der Assets — Schlüssel im Ergebnis-Dict
    // z:            fertige korrelierte Zufallszahlen [n_sims, n_steps, total_noise_dim]
    // dt:           Zeitschrittgröße
    // Gibt zurück:  dict { asset_id → ndarray }
    //               GBM:    [n_sims, n_steps]
    //               Heston: [n_sims, n_steps]  — nur Spot, Varianz intern
    static pybind11::dict generate(
        const std::vector<pybind11::dict>& params_list,
        const std::vector<std::string>& asset_ids,
        const pybind11::array_t<double>& z,
        double dt
    );

private:
    // Erzeugt den richtigen Prozess aus einem Parameter-Dict
    static std::unique_ptr<ProcessBase> build_process(
        const pybind11::dict& params
    );

    // Simuliert einen einzelnen Prozess über alle Zeitschritte
    // z_asset: Zufallszahlen für dieses Asset [n_sims, n_steps, noise_dim]
    // Gibt Spot-Pfad zurück: [n_sims, n_steps]
    static pybind11::array_t<double> simulate_process(
        const ProcessBase& process,
        const pybind11::array_t<double>& z_asset,
        double dt,
        int n_sims,
        int n_steps
    );
};

} // namespace mc