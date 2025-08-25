# TT-Forge-FE Dependency Management

This document explains how dependencies are managed in TT-Forge-FE for both Docker and non-Docker workflows.

## Overview

TT-Forge-FE uses a two-tier dependency system:

1. **Core Dependencies**: Essential packages required for the main functionality (installed with the core package)
2. **Model Dependencies**: Additional packages required for specific model tests (installed separately)

This separation allows for:
- Lightweight core package installation for users who don't need all model capabilities
- Faster Docker builds with all dependencies pre-installed
- Flexible dependency management for different use cases

## For Docker Users

When building Docker images, all model dependencies are automatically installed during the build process. No additional action is required.

The Docker build process:
1. Installs core dependencies via `setup.py`
2. Automatically discovers and installs all model-specific requirements from `forge/test/models/**/requirements.txt` files
3. Creates a complete environment ready for all model tests

## For Non-Docker Users

### Quick Start (Recommended)

Install everything with a single command:

```bash
./scripts/install_deps.sh
```

This will:
1. Install the core TT-Forge-FE package with minimal dependencies
2. Install all model-specific dependencies

### Granular Control

If you want more control over what gets installed:

```bash
# Install only the core package (minimal dependencies)
./scripts/install_deps.sh --core-only

# Install only model dependencies (assumes core package is already installed)
./scripts/install_deps.sh --models-only

# Install everything (same as default behavior)
./scripts/install_deps.sh --all
```

### Manual Installation

For advanced users who prefer manual control:

```bash
# Install core package
pip install -e .

# Install model dependencies
python scripts/install_model_deps.py

# Or just see what would be installed without installing
python scripts/install_model_deps.py --dry-run

# Generate a requirements file for model dependencies
python scripts/install_model_deps.py --requirements-file model_requirements.txt
```

## Adding New Model Dependencies

To add dependencies for a new model:

1. Create a `requirements.txt` file in your model's directory:
   ```
   forge/test/models/your_framework/your_model/requirements.txt
   ```

2. List the required packages in the file:
   ```
   your-model-package==1.0.0
   additional-dependency>=2.0.0
   ```

3. The dependency will be automatically discovered and installed:
   - During Docker image builds
   - When running `scripts/install_model_deps.py`
   - When using `scripts/install_deps.sh`

## Dependency Conflict Resolution

The model dependency collector handles conflicts intelligently:

- **Same package, same version**: Duplicate entries are ignored
- **Same package, one versioned, one unversioned**: The unversioned (latest) requirement is preferred
- **Same package, different versions**: An error is raised to prevent conflicts

## Architecture Details

### Files Involved

- `env/core_requirements.txt`: Core dependencies for the main package
- `env/linux_requirements.txt`: Linux-specific dependencies
- `forge/test/models/**/requirements.txt`: Model-specific dependencies
- `setup.py`: Simplified to only include core and Linux dependencies
- `scripts/install_model_deps.py`: Model dependency collection and installation
- `scripts/install_deps.sh`: Convenient wrapper for non-Docker users
- `.github/Dockerfile.ci`: Modified to install model dependencies during build

### Workflow Comparison

| Workflow | Core Dependencies | Model Dependencies | Command |
|----------|-------------------|-------------------|---------|
| Docker | ✓ (automatic) | ✓ (automatic) | `docker build` |
| Non-Docker (full) | ✓ | ✓ | `./scripts/install_deps.sh` |
| Non-Docker (minimal) | ✓ | ✗ | `pip install -e .` |
| Non-Docker (models only) | ✗ | ✓ | `python scripts/install_model_deps.py` |

## Migration from Previous System

If you were previously relying on `setup.py` to install all dependencies:

- **Docker users**: No change required, everything is automatic
- **Non-Docker users**: Use `./scripts/install_deps.sh` instead of just `pip install -e .` for the full environment

The old `pip install -e .` command will still work but will only install core dependencies.
