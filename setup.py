# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import re
import sys
import sysconfig
import platform
import subprocess

__requires__ = ['pip >= 24.0']

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


forge_files = {
    "test" : {
        "path": "forge/test",
        "files": [
            "conftest.py",
            "__init__.py",
            "utils.py",
            "test_user.py"
        ]
    }
}

class TTExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])

class MyBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            build_lib = self.build_lib
            if not os.path.exists(build_lib):
                continue # editable install?

            # Build using our make flow, and then copy the file over

            # Pass the required variables for building Wormhole or Grayskull
            if "BACKEND_ARCH_NAME" not in os.environ:
                print("Please provide environment variable `BACKEND_ARCH_NAME` to the build process.")
                sys.exit(1)

            additional_env_variables = {
                "BACKEND_ARCH_NAME": os.environ.get("BACKEND_ARCH_NAME"),
            }
            env = os.environ.copy()
            env.update(additional_env_variables)
            nproc = os.cpu_count()
            subprocess.check_call(["make", f"-j{nproc}", "forge", r'DEVICE_VERSIM_INSTALL_ROOT=\$$ORIGIN/../..'], env=env)

            src = "build/lib/libforge_csrc.so"
            self.copy_file(src, os.path.join(build_lib, filename))

            self._copy_forge(build_lib)

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
with open("python_env/core_requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Add specific requirements for distribution
# due to how we use the requirements file we can not use include requirements files
with open("python_env/dist_requirements.txt", "r") as f:
    requirements += [r for r in f.read().splitlines() if not r.startswith("-r")]

# forge._C
forge_c = TTExtension("forge._C")


ext_modules = [forge_c]

packages = [p for p in find_packages("forge") if not p.startswith("test")]

short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
date = subprocess.check_output(['git', 'show', '-s', '--format=%cd', "--date=format:%y%m%d", 'HEAD']).decode('ascii').strip()

arch_codes = {"wormhole_b0": "wh_b0", "grayskull": "gs", "blackhole": "bh"}
arch_code = arch_codes[os.environ["BACKEND_ARCH_NAME"]]

version = "0.1." + date + "+dev." + arch_code + "." + short_hash

setup(
    name='forge',
    version=version,
    author='Tenstorrent',
    url="http://www.tenstorrent.com",
    author_email='info@tenstorrent.com',
    description='AI/ML framework for Tenstorrent devices',
    python_requires='>=3.8',
    packages=packages,
    package_data={"forge": ["tti/runtime_param_yamls/*.yaml"]},
    package_dir={"forge": "forge/forge"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=MyBuild),
    zip_safe=False,
    install_requires=requirements,
    license="TBD",
    keywords="forge machine learning tenstorrent",
    # PyPI
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)
