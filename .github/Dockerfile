FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
ENV TTFORGE_TOOLCHAIN_DIR=/opt/ttforge-toolchain
ENV PROJECT_NAME=tt-forge
ARG GITHUB_TOKEN

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    python3-dev \
    python3-venv \
    python3-pip \
    git \
    git-lfs \
    libhwloc-dev \
    pandoc \
    libtbb-dev \
    libcapstone-dev \
    pkg-config \
    linux-tools-generic \
    ninja-build \
    wget \
    cmake \
    ccache \
    doxygen \
    libgtest-dev \
    libgmock-dev \
    graphviz \
    patchelf \
    libyaml-cpp-dev \
    libboost-all-dev

# Install clang 17
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod u+x llvm.sh && \
    ./llvm.sh 17 && \
    apt install -y libc++-17-dev libc++abi-17-dev && \
    ln -s /usr/bin/clang-17 /usr/bin/clang && \
    ln -s /usr/bin/clang++-17 /usr/bin/clang++

# Install python packages
RUN pip install cmake \
    pytest

# Create a directory for the build and toolchain
ARG BUILD_DIR=/home/build
RUN mkdir -p $BUILD_DIR && \
    mkdir -p $TTMLIR_TOOLCHAIN_DIR && \
    mkdir -p $TTFORGE_TOOLCHAIN_DIR

# Clone the project and update submodules
# Pass in PAT to clone private repositories
RUN git clone https://$GITHUB_TOKEN@github.com/tenstorrent/tt-forge.git $BUILD_DIR/$PROJECT_NAME && \
    cd $BUILD_DIR/$PROJECT_NAME && \
    git submodule update --init --recursive -f

# Build the toolchain
WORKDIR $BUILD_DIR/$PROJECT_NAME
RUN source env/activate && \
    cmake -B env/build env && \
    cmake --build env/build

# Build project to test the container
RUN source env/activate && \
    cmake -G Ninja -B build . && \
    cmake --build build

# Clean up the build directory
RUN cd .. && \
    rm -rf $BUILD_DIR/$PROJECT_NAME && \
    unset GITHUB_TOKEN