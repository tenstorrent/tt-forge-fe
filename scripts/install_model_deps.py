#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Model Dependencies Installation Script

This script collects and installs model-specific Python package requirements
from all `requirements.txt` files under the forge/test/models directory.

Usage:
    python scripts/install_model_deps.py [--dry-run] [--requirements-file OUTPUT_FILE]

Options:
    --dry-run           Show what would be installed without actually installing
    --requirements-file Generate a consolidated requirements file instead of installing
    --help              Show this help message
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


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
    # Valid package names: letters, numbers, hyphens, underscores, dots (PEP 508 compliant)
    # Valid version specifiers: complex patterns like >=1.0.0,<2.0.0
    version_pattern = re.compile(r"^([a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)(.+)?$")

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
                pkg_name, _, version = match.groups()

                # Additional validation for package name (must be PEP 508 compliant)
                if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$", pkg_name):
                    raise ValueError(f"Invalid package name: '{pkg_name}' in {req_file}")

                # If version is None (e.g., just "torch"), treat it as empty string
                version = version or ""

                # Additional validation for version specifier (if present)
                # Valid version specifiers: ==1.0.0, >=1.0.0, >=1.0.0,<2.0.0, ~=1.0.0, etc.
                if version and not re.match(
                    r"^[<>=!~]*[0-9a-zA-Z.,\s*+-]+([,\s]*[<>=!~]+[0-9a-zA-Z.,\s*+-]+)*$", version
                ):
                    raise ValueError(f"Invalid version specifier: '{version}' for package '{pkg_name}' in {req_file}")

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


def install_requirements(requirements: list[str], dry_run: bool = False) -> None:
    """
    Install the given requirements using pip.

    Args:
        requirements: List of requirement strings to install
        dry_run: If True, only print what would be installed without actually installing
    """
    if not requirements:
        print("No model requirements found.")
        return

    if dry_run:
        print("Would install the following packages:")
        for req in requirements:
            print(f"  - {req}")
        return

    print(f"Installing {len(requirements)} model dependencies...")
    cmd = [sys.executable, "-m", "pip", "install"] + requirements

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Successfully installed model dependencies!")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install model dependencies: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        sys.exit(1)


def write_requirements_file(requirements: list[str], output_file: str) -> None:
    """
    Write the collected requirements to a file.

    Args:
        requirements: List of requirement strings to write
        output_file: Path to the output file
    """
    with open(output_file, "w") as f:
        f.write("# Model-specific requirements collected from forge/test/models\n")
        f.write("# Generated by scripts/install_model_deps.py\n\n")
        for req in requirements:
            f.write(f"{req}\n")

    print(f"Requirements written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Install model-specific dependencies for TT-Forge-FE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be installed without actually installing"
    )
    parser.add_argument(
        "--requirements-file",
        metavar="OUTPUT_FILE",
        help="Generate a consolidated requirements file instead of installing",
    )

    args = parser.parse_args()

    # Determine the root directory for model requirements
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_requirements_root = project_root / "forge" / "test" / "models"

    if not model_requirements_root.exists():
        print(f"Error: Model requirements directory not found: {model_requirements_root}")
        sys.exit(1)

    print(f"Collecting model requirements from: {model_requirements_root}")

    try:
        requirements = collect_model_requirements(str(model_requirements_root))

        if args.requirements_file:
            write_requirements_file(requirements, args.requirements_file)
        else:
            install_requirements(requirements, args.dry_run)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
