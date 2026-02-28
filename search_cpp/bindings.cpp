#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

std::string dummy_backend_name() {
    return "rainbow_search_cpp";
}

PYBIND11_MODULE(rainbow_search_cpp, m) {
    m.doc() = "Dummy pybind11 module for Rainbow search backend";
    m.def("dummy_backend_name", &dummy_backend_name, "Return the active C++ backend name");
}
