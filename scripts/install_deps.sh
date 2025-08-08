#!/bin/bash
# TT-Forge-FE Dependency Installation Script
#
# This script provides an easy way for non-Docker users to install
# the core tt-forge-fe package and all additional model dependencies.
#
# Usage:
#   ./scripts/install_deps.sh [--core-only] [--models-only] [--all]
#
# Options:
#   --core-only    Install only the core package (default pip install behavior)
#   --models-only  Install only the model-specific dependencies
#   --all          Install both core package and model dependencies (default)
#   --help         Show this help message

set -e  # Exit on any error

# Default behavior
INSTALL_CORE=true
INSTALL_MODELS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --core-only)
            INSTALL_CORE=true
            INSTALL_MODELS=false
            shift
            ;;
        --models-only)
            INSTALL_CORE=false
            INSTALL_MODELS=true
            shift
            ;;
        --all)
            INSTALL_CORE=true
            INSTALL_MODELS=true
            shift
            ;;
        --help|-h)
            echo "TT-Forge-FE Dependency Installation Script"
            echo ""
            echo "Usage: $0 [--core-only] [--models-only] [--all] [--help]"
            echo ""
            echo "Options:"
            echo "  --core-only    Install only the core package (minimal dependencies)"
            echo "  --models-only  Install only the model-specific dependencies"
            echo "  --all          Install both core package and model dependencies (default)"
            echo "  --help         Show this help message"
            echo ""
            echo "For Docker users, model dependencies are automatically installed during image build."
            echo "Non-Docker users can use this script for easy dependency management."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

echo "TT-Forge-FE Dependency Installation"
echo "==================================="

if [ "$INSTALL_CORE" = true ]; then
    echo "Installing core package with minimal dependencies..."
    pip install -e .
    echo "✓ Core package installed successfully!"
fi

if [ "$INSTALL_MODELS" = true ]; then
    echo "Installing model-specific dependencies..."
    python scripts/install_model_deps.py
    echo "✓ Model dependencies installed successfully!"
fi

echo ""
echo "Installation complete!"
if [ "$INSTALL_CORE" = true ] && [ "$INSTALL_MODELS" = true ]; then
    echo "You now have the full TT-Forge-FE environment ready for all model tests."
elif [ "$INSTALL_CORE" = true ]; then
    echo "You have the core TT-Forge-FE package installed."
    echo "Run '$0 --models-only' later to install model-specific dependencies."
elif [ "$INSTALL_MODELS" = true ]; then
    echo "Model-specific dependencies have been installed."
    echo "Make sure you have the core package installed with 'pip install -e .' if needed."
fi
