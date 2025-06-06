# Building

This document describes how to build the tt-forge-fe project on your local machine.

## Prerequisites
The main project dependencies are:
1. Clang 17
2. Ninja
3. CMake 3.20 or higher
4. Python 3.10 or higher

On Ubuntu 22.04 systems, you can install these dependencies using the following commands:
```sh
# Update package list
sudo apt update -y
sudo apt upgrade -y

# Install Clang
sudo apt install clang-17

# Install Ninja
sudo apt install ninja-build

# Install CMake
sudo apt remove cmake -y
pip3 install cmake --upgrade
cmake --version

# Check Python version
python3 --version
```

## Building the Environment
This is a one off step to build the toolchain and create a virtual environment for `tt-forge-fe`. Generally, you need to run this step only once, unless you want to update the toolchain. Since `tt-forge-fe` is using `tt-mlir`, this step also builds the `tt-mlir` environment (toolchain).

First, it's required to create toolchain directories. The proposed example creates directories using the default paths. You can change the paths if you want to use different locations (see the [Useful Build Environment Variables](#useful-build-environment-flags) section below).
```sh
# FFE related toolchain (dafault path)
sudo mkdir -p /opt/ttforge-toolchain
sudo chown -R $USER /opt/ttforge-toolchain

# MLIR related toolchain (default path)
sudo mkdir -p /opt/ttmlir-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain
```

Build FFE environment:
```sh
# Initialize required env vars
source env/activate

# Initialize and update submodules
git submodule update --init --recursive -f

# Build environment
cmake -B env/build env
cmake --build env/build
```

> **Expert Tip:** If you already have the `tt-mlir` toolchain built, you can use the `TTFORGE_SKIP_BUILD_TTMLIR_ENV` option to skip rebuilding the `tt-mlir` environment (toolchain) to save time.
> ```sh
> cmake -B env/build env -DTTFORGE_SKIP_BUILD_TTMLIR_ENV=ON
> cmake --build env/build
> ```
>
> **NOTE:** special care should be taken to ensure that the already built `tt-mlir` environment (toolchain) version is compatible with the one `tt-forge-fe` is using.

## Build Forge
```sh
# Activate virtual environment
source env/activate

# Build Forge
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
cmake --build build
```

> **NOTE:** our official compiler is `clang-17`, building with other compilers is at the moment not tested.
>
> If you want to try other compilers, you can do so by changing the `-DCMAKE_CXX_COMPILER` and `-DCMAKE_C_COMPILER` options.

You can pass additional options to the `cmake` command to customize the build. For example, to build everything in debug mode, you can run:
```sh
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
cmake --build build
```

> List of commonly used options:
> - `-DCMAKE_BUILD_TYPE=Debug|Release`      - Build type (Debug, Release)
> - `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache`  - Use [`ccache`](https://ccache.dev/) to speed up re-builds
> - `-DTTMLIR_RUNTIME_DEBUG=ON|OFF`         - Build runtime debug tools (more logging, debug environment flags)

### Incremental Building
If you have made changes to the C++ sources (of the `tt-forge-fe` compiler, `tt-mlir` or `tt-metal`), you might want to do an incremental build to save time. This can be done by running the following command:
```sh
# If you are not already inside the virtual environment, activate it
source env/activate

cmake --build build -- install_ttforge
```

This will build `tt-forge-fe` C++ sources and the dependencies (`tt-mlir`, `tt-metal`) and install them in the virtual environment.

## Building the Docs

To build documentation `mdbook` is required, see the installation guide [here](./tools.md#mdbook).

After installing `mdbook`, run the following commands to build and serve the documentation:

```sh
source env/activate
cmake --build build -- docs

# Serve the documentation
mdbook serve build/docs
```

> **Note:** `mdbook serve` will by default create a local server at `http://localhost:3000`.

> **Note:** For a custom port, specify the `-p` attribute. <br><br> E.g. `mdbook serve build/docs -p 5005`, and visit `http://localhost:5005`.

## Build Cleanup

To ensure a clean build environment, follow these steps to remove existing build artifacts:

1. Remove tt-forge-fe build artifacts:
    ```sh
    rm -rf build
    ```
    > **Note:** This command removes the `build` directory and all its contents, effectively cleaning up the build artifacts specific to tt-forge-fe.

2. Clean all tt-forge-fe build artifacts:
     ```sh
     ./clean_build.sh
     ```
   > **Note:** This script executes a comprehensive cleanup, removing all build artifacts across the entire Forge project, ensuring a clean slate for subsequent builds.

   > **Note:** The `clean_build.sh` script will not clean toolchain (LLVM) build artifacts and dependencies.

3. Clean everything (including the environment):
    ```sh
    ./clean_build.sh
    rm -rf env/build third_party/tt-mlir/env/build
    ```
    > **Note:** This should rarely be needed, as it removes the entire build and environment (consequently entire toolchain will need to be rebuilt).

## Useful Build Environment Variables
1. `TTMLIR_TOOLCHAIN_DIR` - Specifies the directory where TTMLIR dependencies will be installed. Defaults to `/opt/ttmlir-toolchain` if not defined.
2. `TTMLIR_VENV_DIR` - Specifies the virtual environment directory for TTMLIR. Defaults to `/opt/ttmlir-toolchain/venv` if not defined.
3. `TTFORGE_TOOLCHAIN_DIR` - Specifies the directory where tt-forge dependencies will be installed. Defaults to `/opt/ttforge-toolchain` if not defined.
4. `TTFORGE_VENV_DIR` - Specifies the virtual environment directory for tt-forge. Defaults to `/opt/ttforge-toolchain/venv` if not defined.
5. `TTFORGE_PYTHON_VERSION` - Specifies the Python version to use. Defaults to `python3.10` if not defined.

## Run tt-forge-fe Using a Docker Image

We provide two Docker images for tt-forge-fe:

1. **Base Image**: This image includes all the necessary preinstalled dependencies.
2. **Prebuilt Environment Image**: This image also comes with a prebuilt environment, allowing you to skip the environment build step.

```sh
ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-base-ird-ubuntu-22-04
ghcr.io/tenstorrent/tt-forge-fe/tt-forge-fe-ird-ubuntu-22-04
```

> **Note:** To be able to build tt-forge-fe inside the docker containers, make sure to set yourself as the owner of the tt-forge-fe and tt-mlir toolchain directories -
```sh
sudo chown -R $USER /opt/ttforge-toolchain
sudo chown -R $USER /opt/ttmlir-toolchain
```
