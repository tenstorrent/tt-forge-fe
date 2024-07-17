1. Build environment
    This will download all dependencies needed for pybuda.
    source env/activate - To load the environment variables for the project.
    cmake -B env/build env
    cmake --build env/build

2. Build pybuda
    source env/activate - To load the environment variables for the project and activate the virtual environment.
    cmake -B build .
    cmake -G Ninja --build build

3. Running pybuda
    source env/activate

Environment variables:
    TTMLIR_TOOLCHAIN_DIR - points to toolchain dir where dependencies of TTLMIR will be installed. If not defined 
    it defaults to /opt/ttmlir-toolchain
    TTMLIR_VENV_DIR - points to virtual environment directory of TTMLIR.If not defined
    it defaults to /opt/ttmlir-toolchain/venv
    PYBUDA_TOOLCHAIN_DIR - points to toolchain dir where dependencies of PyBuda will be installed. If not defined
    it defaults to /opt/pybuda-toolchain
    PYBUDA_VENV_DIR - points to virtual environment directory of Pybuda. If not defined
    it defaults to /opt/pybuda-toolchain/venv
