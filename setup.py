# setup.py
from setuptools import setup, find_packages, Extension
import pybind11

ext = Extension(
    name="mc_engine.mc_core",          # Import als: from mc_engine import mc_core
    sources=[
        "cpp/src/processes/gbm.cpp",
        "cpp/src/processes/heston.cpp",
        "cpp/src/processes/vasicek.cpp",
        "cpp/src/processes/gamma_variance.cpp",
        "cpp/src/processes/cir.cpp",
        "cpp/src/path_generator.cpp",
        "cpp/src/utils/cholesky.cpp",
        "cpp/bindings/bindings.cpp",
    ],
    include_dirs=[
        "cpp/include",
        pybind11.get_include(),
    ],
    extra_compile_args=["-std=c++17", "-march=native"],
    language="cpp",
)

setup(
    packages=find_packages(include=["mc_engine", "mc_engine.*"]),
    ext_modules=[ext],
)