#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import shutil


def manage_pyproject_toml(disable=False):
    """Temporarily disable/enable pyproject.toml to allow setup.py full control"""
    if disable:
        if os.path.exists("pyproject.toml"):
            if os.path.exists("pyproject.toml.backup"):
                os.remove("pyproject.toml.backup")
            shutil.move("pyproject.toml", "pyproject.toml.backup")
    else:
        if os.path.exists("pyproject.toml.backup"):
            if os.path.exists("pyproject.toml"):
                os.remove("pyproject.toml")
            shutil.move("pyproject.toml.backup", "pyproject.toml")


def clean_build_dir():
    """Clean build directory to prevent conflicts between builds"""
    if os.path.exists("build"):
        shutil.rmtree("build")


def build_and_collect_wheels():

    try:
        # Temporarily disable pyproject.toml
        manage_pyproject_toml(disable=True)

        # Build compiler variant
        clean_build_dir()
        os.environ["FORGE_VARIANT"] = "compiler"
        subprocess.run(["python3", "setup.py", "bdist_wheel"], check=True)

        # Build dev variant
        clean_build_dir()
        os.environ["FORGE_VARIANT"] = "dev"
        subprocess.run(["python3", "setup.py", "bdist_wheel"], check=True)

        built_files = os.listdir("dist")

        # Collect compiler wheel
        compiler_wheel = None
        dev_wheel = None

        for fname in built_files:
            if fname.startswith("tt_forge_fe_compiler") and fname.endswith(".whl"):
                compiler_wheel = fname
            elif fname.startswith("tt_forge_fe_dev") and fname.endswith(".whl"):
                dev_wheel = fname

        # Move compiler wheel
        if compiler_wheel:
            os.makedirs("dist/compiler", exist_ok=True)
            shutil.move(os.path.join("dist", compiler_wheel), os.path.join("dist/compiler", compiler_wheel))

        # Move dev wheel
        if dev_wheel:
            os.makedirs("dist/dev", exist_ok=True)
            shutil.move(os.path.join("dist", dev_wheel), os.path.join("dist/dev", dev_wheel))

    finally:
        # Always restore pyproject.toml
        manage_pyproject_toml(disable=False)


# Make sure clean slate
os.makedirs("dist", exist_ok=True)

# Build both wheels
build_and_collect_wheels()
