### Building dependencies
* `cmake`
* `clang`
* `Ninja` - sudo apt-get install ninja-build

### Building environment
This is one off step. It will pull all dependencies needed for tt-forge.

* `git submodule update --init --recursive -f`
* `source env/activate`
* `cmake -B env/build env`
* `cmake --build env/build`

### Build tt-forge
* `source env/activate`
* `cmake -G Ninja --build build .`
* `cmake --build build`

### Cleanup
* `rm -rf build` - to cleanup tt-forge build artifacts.
* `./clean_all.sh` - to cleanup all build artifacts (tt-forge/tvm/tt-mlir/tt-metal). This will not remove toolchain dependencies.

### Environment variables:
* `TTMLIR_TOOLCHAIN_DIR` - points to toolchain dir where dependencies of TTLMIR will be installed. If not defined it defaults to /opt/ttmlir-toolchain
* `TTMLIR_VENV_DIR` - points to virtual environment directory of TTMLIR.If not defined it defaults to /opt/ttmlir-toolchain/venv
* `PYBUDA_TOOLCHAIN_DIR` - points to toolchain dir where dependencies of PyBuda will be installed. If not defined it defaults to /opt/pybuda-toolchain
* `PYBUDA_VENV_DIR` - points to virtual environment directory of tt-forge. If not defined it defaults to /opt/pybuda-toolchain/venv
* `PYBUDA_PYTHON_VERSION` - set to override python version. If not defined it defaults to python3.10
