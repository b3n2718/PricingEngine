#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "path_generator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mc_core, m) {
    m.doc() = "Monte Carlo Path Generation Core";

    m.def("generate",
          &mc::PathGenerator::generate,
          py::arg("params_list"),
          py::arg("asset_ids"),
          py::arg("z"),
          py::arg("dt"),
          R"doc(
            Simuliert Preispfade gegeben fertige Zufallszahlen.

            Parameters
            ----------
            params_list : list[dict]  — Prozessparameter, eines pro Asset
            asset_ids   : list[str]   — Asset-Bezeichner
            z           : ndarray     — korrelierte N(0,1) Zahlen [n_sims, n_steps, total_noise_dim]
            dt          : float       — Zeitschrittgröße

            Returns
            -------
            dict { asset_id -> ndarray[n_sims, n_steps] }
          )doc"
    );
}