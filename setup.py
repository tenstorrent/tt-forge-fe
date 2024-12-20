# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import re
import sys
import sysconfig
import platform
import subprocess

__requires__ = ["pip >= 24.0"]

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


forge_files = {"test": {"path": "forge/test", "files": ["conftest.py", "__init__.py", "utils.py", "test_user.py"]}}


class TTExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:

            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            print(f"Building {filename}")

            print(f"Building {fullname}")
            extension_path = pathlib.Path(self.get_ext_fullpath(ext.name))
            print(f"Extension path: {extension_path}")

            build_lib = self.build_lib
            if not os.path.exists(build_lib):
                continue  # editable install?

            # Build using our make flow, and then copy the file over
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
            #
            # self.copy_tree(install_dir / "lib", build_lib + "/forge", preserve_symlinks=False)

    def _copy_forge(self, target_path):

        for t, d in forge_files.items():
            path = target_path + "/" + d["path"]
            os.makedirs(path, exist_ok=True)

            src_path = d["path"]
            if d["files"] == "*":
                self.copy_tree(src_path, path)
            else:
                for f in d["files"]:
                    self.copy_file(src_path + "/" + f, path + "/" + f)


with open("README.md", "r") as f:
    long_description = f.read()

# Read the requirements from the core list that is shared between
# dev and test.
with open("env/core_requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Add specific requirements for distribution
# due to how we use the requirements file we can not use include requirements files
with open("env/linux_requirements.txt", "r") as f:
    requirements += [r for r in f.read().splitlines() if not r.startswith("-r")]

# forge._C
forge_c = TTExtension("libttforge_csrc.so")


ext_modules = [forge_c]

packages = [p for p in find_packages("forge") if not p.startswith("test")]

short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
date = (
    subprocess.check_output(["git", "show", "-s", "--format=%cd", "--date=format:%y%m%d", "HEAD"])
    .decode("ascii")
    .strip()
)

version = "0.1." + date + "+dev." + short_hash

setup(
    name="tt-forge-fe",
    version=version,
    author="Tenstorrent",
    url="https://github.com/tenstorrent/tt-forge-fe",
    author_email="info@tenstorrent.com",
    description="AI/ML framework for Tenstorrent devices",
    python_requires=">=3.8",
    packages=packages,
    package_data={"forge": ["tti/runtime_param_yamls/*.yaml"]},
    package_dir={"forge": "forge/forge"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=requirements,
    license="TBD",
    keywords="forge machine learning tenstorrent",
    # PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
