# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import subprocess
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

forge_files = {"test": {"path": "forge/test", "files": ["conftest.py", "__init__.py", "utils.py", "test_user.py"]}}


class TTExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "forge" in ext.name:
                self.build_forge(ext)
            elif "tvm" in ext.name:
                self.build_tvm(ext)

    def build_tvm(self, ext):
        pass
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)
        print(f"Building {filename}")
        print(f"Building {fullname}")
        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Extension path: {extension_path}")

        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            # Might be an editable install or something else
            return

        cwd = pathlib.Path().absolute()
        build_dir = cwd / "wheel" / "build"
        install_dir = cwd / "wheel" / "install"

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ]

        self.spawn(["cmake", *cmake_args])
        self.spawn(["cmake", "--build", str(build_dir)])
        self.spawn(["cmake", "--install", str(build_dir)])

        print(f"Copying {install_dir} to {build_lib}")
        self.copy_tree(str(install_dir), str(build_lib))

    def build_forge(self, ext):
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)
        print(f"Building {filename}")
        print(f"Building {fullname}")
        extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        print(f"Extension path: {extension_path}")

        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            # Might be an editable install or something else
            return

        cwd = pathlib.Path().absolute()
        build_dir = cwd / "wheel" / "build"
        install_dir = cwd / "wheel" / "install"

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DTTMLIR_RUNTIME_DEBUG=OFF",
        ]

        self.spawn(["cmake", *cmake_args])
        self.spawn(["cmake", "--build", str(build_dir)])
        self.spawn(["cmake", "--install", str(build_dir)])

        self.copy_tree(str(install_dir), str(extension_path.parent))


with open("README.md", "r") as f:
    long_description = f.read()
# Print current directory
print(f"Current directory: {os.getcwd()}")
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

# Find packages as before
packages = [p for p in find_packages("forge") if not p.startswith("test")]

setup(
    name="tt-forge-fe",
    version=version,
    install_requires=requirements,
    packages=packages,
    package_dir={"forge": "forge/forge"},
    ext_modules=[forge_c],
    cmdclass={"build_ext": CMakeBuild},
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
)
