# tt-forge
tt-forge is our front end compiler project aimed at utitlizing our latest [tt-mlir](https://github.com/tenstorrent/tt-mlir) project. This is a continuation of our work in [tt-buda](https://github.com/tenstorrent/tt-buda) and is built using some of the same functionality from that compiler.

Project goals
-----
- Remove the [BBE](https://github.com/tenstorrent/tt-budabackend) requirements and link against tt-mlir
- emit a graph as MLIR


Building dependencies
-----
* `cmake3.20`
* `clang` and `clang++-17`
* `Ninja`
* `Python3.10`

Building environment
-----
This is one off step to build the toolchain and create virtual environment for tt-forge.
Generally you need to run this step only once, unless you want to update the toolchain.

* `git submodule update --init --recursive -f`
* `source env/activate`
* `cmake -B env/build env`
* `cmake --build env/build`

Build tt-forge
-----
* `source env/activate`
* `cmake -G Ninja -B build .`
* `cmake --build build`

Cleanup
-----
* `rm -rf build` - to cleanup tt-forge build artifacts.
* `./clean_all.sh` - to cleanup all build artifacts (tt-forge/tvm/tt-mlir/tt-metal). This will not remove toolchain dependencies.

Environment variables:
-----
* `TTMLIR_TOOLCHAIN_DIR` - points to toolchain dir where dependencies of TTLMIR will be installed. If not defined it defaults to /opt/ttmlir-toolchain
* `TTMLIR_VENV_DIR` - points to virtual environment directory of TTMLIR.If not defined it defaults to /opt/ttmlir-toolchain/venv
* `TTFORGE_TOOLCHAIN_DIR` - points to toolchain dir where dependencies of tt-forge will be installed. If not defined it defaults to /opt/ttforge-toolchain
* `TTFORGE_VENV_DIR` - points to virtual environment directory of tt-forge. If not defined it defaults to /opt/ttforge-toolchain/venv
* `TTFORGE_PYTHON_VERSION` - set to override python version. If not defined it defaults to python3.10
