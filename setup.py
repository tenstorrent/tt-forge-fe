# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pathlib
import subprocess
import sys
from datetime import datetime
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from pathlib import Path
import re
import urllib.request


class TTExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if "forge" in ext.name:
                self.build_forge(ext)
            else:
                raise Exception("Unknown extension")

    def build_forge(self, ext):
        build_lib = self.build_lib
        if not os.path.exists(build_lib):
            # Might be an editable install or something else
            return

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
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DTTMLIR_RUNTIME_DEBUG=OFF",
            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
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


def collect_model_requirements(requirements_root: str) -> list[str]:
    """
    Collect and deduplicate model-specific Python package requirements from all `requirements.txt` files
    under the given root directory.

    Handles version conflicts as follows:
    - If the same package appears with different versions, an error is raised.
    - If one occurrence has a version and another does not, the no-version spec (latest) is preferred.
    - Duplicate entries with the same version are ignored.

    Args:
        requirements_root (str): Path to the directory to search for requirements.txt files.

    Returns:
        List[str]: A sorted list of unique requirement strings (e.g., ["numpy>=1.21", "torch"]).
    """

    # Regex to capture package name and optional version specifier
    # e.g., "torch>=2.1.0" → ("torch", ">=2.1.0")
    version_pattern = re.compile(r"^([a-zA-Z0-9_\-]+)([<>=!~]+.+)?$")

    # Tracks source of each package → {package_name: (version_str, file_path)}
    requirement_map = {}

    # Final deduplicated output → {package_name: version_str}
    # This ensures no duplicates and consistent overwrite behavior
    final_requirements = {}

    # Find all requirements.txt files under the given root directory
    for req_file in Path(requirements_root).rglob("requirements.txt"):

        # Open and read each requirements.txt file
        with open(req_file, "r") as f:
            for line in f:
                line = line.strip()  # Remove whitespace at start/end

                # Skip empty lines or comments
                if not line or line.startswith("#"):
                    continue

                # Extract package name and version using regex
                match = version_pattern.match(line)
                if not match:
                    raise ValueError(f"Unrecognized requirement format: '{line}' in {req_file}")

                # Extract matched groups
                pkg_name, version = match.groups()

                # If version is None (e.g., just "torch"), treat it as empty string
                version = version or ""

                if pkg_name in requirement_map:
                    # We've already seen this package in another file
                    prev_version, prev_file = requirement_map[pkg_name]

                    if prev_version != version:
                        # Conflict: one has version, other has a different version or none

                        if version == "":
                            # Current one has no version → prefer this (more general)
                            requirement_map[pkg_name] = ("", req_file)
                            final_requirements[pkg_name] = ""

                        elif prev_version == "":
                            # Previous one was no version → keep it, ignore current versioned one
                            continue

                        else:
                            # Actual version mismatch → raise an error
                            raise AssertionError(
                                f"Conflicting versions for '{pkg_name}':\n"
                                f"- {prev_version} in {prev_file}\n"
                                f"- {version} in {req_file}"
                            )

                    # else: same version → ignore duplicate
                else:
                    # First time seeing this package → record it
                    requirement_map[pkg_name] = (version, req_file)
                    final_requirements[pkg_name] = version

    # Convert the final dictionary to a list of strings
    # e.g., {"torch": "==2.1.0", "numpy": ""} → ["torch==2.1.0", "numpy"]
    return [pkg + ver if ver else pkg for pkg, ver in sorted(final_requirements.items())]


model_requirements_root = "forge/test/models"
model_requirements = collect_model_requirements(model_requirements_root)

requirements = core_requirements + linux_requirements + model_requirements


# Parse build type from command line arguments
def parse_build_type():
    """
    Parse --build-type argument from command line.
    This is passed from build.yml: python3 setup.py bdist_wheel --build-type release
    Defaults to "release" if not provided.
    """
    build_type = None
    args_to_remove = []

    for i, arg in enumerate(sys.argv):
        if arg == "--build-type":
            if i + 1 < len(sys.argv):
                build_type = sys.argv[i + 1]
                args_to_remove = [i, i + 1]
                break
            else:
                raise ValueError("--build-type argument requires a value")

    # Remove the arguments so setuptools doesn't see them
    for i in reversed(args_to_remove):
        sys.argv.pop(i)

    # Default to Release if not provided
    if build_type is None:
        build_type = "release"

    return build_type.lower()


def get_git_commit_hash(repo_path: str = ".") -> str:
    """Get full git commit hash from a repository path."""
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ValueError(f"Failed to get git commit hash from {repo_path}: {e}")


def get_tt_mlir_commit_hash() -> str:
    """
    Get tt-mlir SHA from the git submodule.
    In tt-forge-fe, tt-mlir is a git submodule, so we get the commit hash directly.
    """
    mlir_path = "third_party/tt-mlir"
    if not os.path.exists(mlir_path):
        raise ValueError(f"tt-mlir submodule not found at {mlir_path}")

    return get_git_commit_hash(mlir_path)


def get_tt_metal_commit_hash() -> str:
    """
    Fetch tt-metal SHA from tt-mlir repo's CMakeLists.txt.
    Matches tt-xla approach: https://github.com/tenstorrent/tt-xla/blob/main/python_package/setup.py#L86
    """

    # Extract tt-metal SHA from third_party/tt-mlir/third_party/CMakeLists.txt 
    cmake_file = pathlib.Path(__file__).resolve().parent / "third_party" / "tt-mlir" / "third_party" / "CMakeLists.txt"
    with cmake_file.open() as f:
        cmake_content = f.read()
    metal_match = re.search(r'set\(TT_METAL_VERSION "([^"]+)"\)', cmake_content)
    if not metal_match:
        raise ValueError("Failed to extract TT_METAL_VERSION from tt-mlir CMakeLists.txt")
    return metal_match.group(1)


def get_build_date() -> str:
    """Get build date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def format_build_summary(build_type: str) -> str:
    """Format build metadata summary string."""
    commit_hash = get_git_commit_hash()
    tt_mlir_commit = get_tt_mlir_commit_hash()
    tt_metal_commit = get_tt_metal_commit_hash()
    built_date = get_build_date()

    return (
        f"commit={commit_hash}, "
        f"tt-mlir-commit={tt_mlir_commit}, "
        f"tt-metal-commit={tt_metal_commit}, "
        f"built-date={built_date}, "
        f"build-type={build_type}"
    )


# Parse build type from command line
build_type = parse_build_type()

# Get build metadata summary
build_summary = format_build_summary(build_type)

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
    name="tt_forge_fe",
    version=version,
    description=build_summary,
    install_requires=requirements,
    packages=packages,
    package_dir={"forge": "forge/forge"},
    ext_modules=[forge_c],
    cmdclass={"build_ext": CMakeBuild},
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
)
