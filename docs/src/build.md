# Building

Following page describes how to build the project on your local machine.

## Prerequisites
Main project dependencies are:
1. Clang 17
1. Ninja
1. CMake 3.20 or higher
1. Git LFS
1. Python 3.10 or higher

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

# Install Git LFS
sudo apt install git-lfs

# Check Python version
python3 --version
```

## Build environment
This is one off step to build the toolchain and create virtual environment for tt-forge. Generally you need to run this step only once, unless you want to update the toolchain (LLVM). 

First, it's required to create toolchain directories. Proposed example creates directories in default paths. You can change the paths if you want to use different locations (see build environment section below). 
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
# Inicialize required env vars
source env/activate

# Initialize and update submodules
git submodule update --init --recursive -f

# Build environment
cmake -B env/build env
cmake --build env/build
```

## Build Forge
```sh
# Activate virtual environment
source env/activate

# Build Forge
cmake -G Ninja -B build
cmake --build build
```

## Build Cleanup

To ensure a clean build environment, follow these steps to remove existing build artifacts:

1. **Clean Forge FE build artifacts**:
    ```sh
    rm -rf build
    ```
   Note: This command removes the `build` directory and all its contents, effectively cleaning up the build artifacts specific to Forge FE.

2. **Clean all Forge build artifacts**:
     ```sh
     ./clean_all.sh
     ```
   Note: This script executes a comprehensive cleanup, removing all build artifacts across the entire Forge project, ensuring a clean slate for subsequent builds.

_Note: `clean_all.sh` script will not clean toolchain (LLVM) build artifacts and dependencies._

## Useful build environment variables
1. `TTMLIR_TOOLCHAIN_DIR` - Specifies the directory where TTMLIR dependencies will be installed. Defaults to `/opt/ttmlir-toolchain` if not defined.
2. `TTMLIR_VENV_DIR` - Specifies the virtual environment directory for TTMLIR. Defaults to `/opt/ttmlir-toolchain/venv` if not defined.
3. `TTFORGE_TOOLCHAIN_DIR` - Specifies the directory where tt-forge dependencies will be installed. Defaults to `/opt/ttforge-toolchain` if not defined.
4. `TTFORGE_VENV_DIR` - Specifies the virtual environment directory for tt-forge. Defaults to `/opt/ttforge-toolchain/venv` if not defined.
5. `TTFORGE_PYTHON_VERSION` - Specifies the Python version to use. Defaults to `python3.10` if not defined.
