# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.build import build


class TTExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class BuildCommand(build):
    def run(self):
        # Ensure that the build directory exists
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Get the list of subcommands which need to be executed.
        # We want to ensure that the `build_ext` sub-command is run before `build_py`.
        subcommands = self.get_sub_commands()
        if "build_ext" in subcommands and "build_py" in subcommands:
            # Move `build_ext` just before `build_py` in self.sub_commands
            # This is a workaround to ensure that `build_ext` runs before `build_py`
            build_ext_index = [subcmd[0] for subcmd in self.sub_commands].index("build_ext")
            build_py_index = [subcmd[0] for subcmd in self.sub_commands].index("build_py")
            if build_ext_index > build_py_index:
                self.sub_commands.insert(build_py_index, self.sub_commands.pop(build_ext_index))

        # Call the parent class's run method
        super().run()


class BuildPyWithMetalPackages(build_py):
    def find_metal_packages(self) -> list[str]:
        # Find all python packages in ttnn - skip test packages.
        file_dir = os.path.dirname(os.path.abspath(__file__))
        ttnn_dir = (
            pathlib.Path(file_dir)
            / "third_party"
            / "tt-mlir"
            / "third_party"
            / "tt-metal"
            / "src"
            / "tt-metal"
            / "ttnn"
        )

        ttnn_packages = find_packages(
            where=ttnn_dir,
            exclude=["ttnn.examples", "ttnn.examples.*", "test"],
        )
        print(f"Found ttnn packages: {ttnn_packages}")

        return ttnn_packages

    def run(self):
        # Add ttnn packages to the list of packages to be built
        ttnn_packages = self.find_metal_packages()
        self.distribution.packages.extend(ttnn_packages)
        self.packages.extend(ttnn_packages)

        # Call the parent class's run method
        super().run()


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "forge" in ext.name:
                self.build_forge(ext)
            else:
                raise Exception("Unknown extension")

    def build_forge(self, ext):
        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Running cmake to install forge at {extension_path}")

        cwd = pathlib.Path().absolute()
        build_dir = cwd / "build"
        install_dir = extension_path.parent / "forge"

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang-17",
            "-DCMAKE_CXX_COMPILER=clang++-17",
            "-DTTMLIR_RUNTIME_DEBUG=OFF",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DTTFORGE_ENABLE_DEVICE_PROFILING=ON",
        ]

        self.spawn(["cmake", *cmake_args])
        self.spawn(["cmake", "--build", str(build_dir)])
        self.spawn(["cmake", "--install", str(build_dir)])


with open("README.md", "r") as f:
    long_description = f.read()

# Compute requirements
with open("env/core_requirements.txt", "r") as f:
    core_requirements = f.read().splitlines()

with open("env/linux_requirements.txt", "r") as f:
    linux_requirements = [r for r in f.read().splitlines() if not r.startswith("-r")]

requirements = core_requirements + linux_requirements

# Compute a dynamic version from git
short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
date = (
    subprocess.check_output(["git", "show", "-s", "--format=%cd", "--date=format:%y%m%d", "HEAD"])
    .decode("ascii")
    .strip()
)
version = "0.1." + date + "+dev." + short_hash

forge_c = TTExtension("forge")

# Find all python packages that we want to install - skip test packages.
packages = [p for p in find_packages("forge") if not p.startswith("test")]
print(f"Found forge packages: {packages}")


# Find all python packages in tt_metal - skip test packages.
# ttmetal_packages = find_packages(
#     where="../tt-forge/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal", exclude=["test"]
# )
# print(f"Found tt_metal packages: {ttmetal_packages}")
#
# packages = packages + ttnn_packages + ttmetal_packages


setup(
    name="forge",
    version=version,
    install_requires=requirements,
    packages=packages,
    package_dir={
        "forge": "forge/forge",
        "ttnn": "../tt-forge/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/ttnn",
        "tracy": "../tt-forge/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/tracy",
        "tt_lib": "../tt-forge/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/tt_lib",
        "tools": "../tt-forge/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/tools",
    },
    ext_modules=[forge_c],
    cmdclass={"build": BuildCommand, "build_ext": CMakeBuild, "build_py": BuildPyWithMetalPackages},
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
)
