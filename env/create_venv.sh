if [[ -z "$TTFORGE_PYTHON_VERSION" ]]; then
    echo "TTFORGE_PYTHON_VERSION environment variable is not set"
    exit 1
fi

if [[ -z "$TTFORGE_TOOLCHAIN_DIR" ]]; then
    echo "TTFORGE_TOOLCHAIN_DIR environment variable is not set"
    exit 1
fi

if [[ -z "$TTFORGE_VENV_DIR" ]]; then
    echo "TTFORGE_VENV_DIR environment variable is not set"
    exit 1
fi

if [[ -z "$CURRENT_SOURCE_DIR" ]]; then
    echo "CURRENT_SOURCE_DIR environment variable is not set"
    exit 1
fi

# Torch requires a specific version of wheel to be installed
# which depends on the platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    REQUIREMENTS_FILE="$CURRENT_SOURCE_DIR/mac_requirements.txt"
else
    # TODO test on linux
    REQUIREMENTS_FILE="$CURRENT_SOURCE_DIR/linux_requirements.txt"
fi

$TTFORGE_PYTHON_VERSION -m venv $TTFORGE_VENV_DIR
unset LD_PRELOAD
source $TTFORGE_VENV_DIR/bin/activate
python -m pip install --upgrade pip
pip3 install wheel==0.37.1
pip3 install -r $REQUIREMENTS_FILE -f https://download.pytorch.org/whl/cpu/torch_stable.html
