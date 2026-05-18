#pragma once
#include <pybind11/numpy.h>
#include <vector>

namespace mc {

namespace py = pybind11;

// process_base.hpp — initial_state hinzufügen


class ProcessBase {
    public:
        virtual ~ProcessBase() = default;
    
        virtual py::array_t<double> evolve(
            const py::array_t<double>& state,
            const py::array_t<double>& z,
            double dt
        ) const = 0;
    
        // NEU — muss hier deklariert sein damit PathGenerator es aufrufen kann
        virtual py::array_t<double> initial_state(int n_sims) const = 0;
    
        virtual int state_dim() const = 0;
        virtual int noise_dim() const = 0;
        std::string type;
    };

} // namespace mc