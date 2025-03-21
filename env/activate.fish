function deactivate -d "Exit virtual environment and return to normal shell environment"
    if test -n "$_OLD_VIRTUAL_PATH"
        set -gx PATH $_OLD_VIRTUAL_PATH
        set -e _OLD_VIRTUAL_PATH
    end
    if test -n "$_OLD_VIRTUAL_PYTHONHOME"
        set -gx PYTHONHOME $_OLD_VIRTUAL_PYTHONHOME
        set -e _OLD_VIRTUAL_PYTHONHOME
    end
    if test -n "$_OLD_FISH_PROMPT_OVERRIDE"
        set -e _OLD_FISH_PROMPT_OVERRIDE
        if functions -q _old_fish_prompt
            functions -e fish_prompt
            functions -c _old_fish_prompt fish_prompt
            functions -e _old_fish_prompt
        end
    end
    if test "$argv[1]" != nondestructive
        set -e TTFORGE_TOOLCHAIN_DIR
        set -e TTFORGE_PYTHON_VERSION
        set -e TTFORGE_VENV_DIR
        set -e TTMLIR_TOOLCHAIN_DIR
        set -e TTMLIR_VENV_DIR
        set -e TTMLIR_ENV_ACTIVATED
        set -e ARCH_NAME
    end
    set -e VIRTUAL_ENV
    set -e VIRTUAL_ENV_PROMPT
    if test "$argv[1]" != nondestructive
        # Self-destruct!
        functions -e deactivate
    end
end

deactivate nondestructive

set -gx TTFORGE_TOOLCHAIN_DIR (test -n "$TTFORGE_TOOLCHAIN_DIR"; and echo "$TTFORGE_TOOLCHAIN_DIR"; or echo "/opt/ttforge-toolchain")
set -gx TTFORGE_PYTHON_VERSION (test -n "$TTFORGE_PYTHON_VERSION"; and echo "$TTFORGE_PYTHON_VERSION"; or echo "python3.10")
set -gx TTFORGE_VENV_DIR (test -n "$TTFORGE_VENV_DIR"; and echo "$TTFORGE_VENV_DIR"; or echo "$TTFORGE_TOOLCHAIN_DIR/venv")
set -gx TTMLIR_TOOLCHAIN_DIR (test -n "$TTMLIR_TOOLCHAIN_DIR"; and echo "$TTMLIR_TOOLCHAIN_DIR"; or echo "/opt/ttmlir-toolchain")
set -gx TTMLIR_VENV_DIR (test -n "$TTMLIR_VENV_DIR"; and echo "$TTMLIR_VENV_DIR"; or echo "$TTMLIR_TOOLCHAIN_DIR/venv")
set -gx TTMLIR_ENV_ACTIVATED 1
set -gx ARCH_NAME (test -n "$ARCH_NAME"; and echo "$ARCH_NAME"; or echo "wormhole_b0")

set -gx VIRTUAL_ENV "$TTFORGE_VENV_DIR"
set -gx _OLD_VIRTUAL_PATH $PATH
set -gx PATH "$TTFORGE_VENV_DIR/bin" "$TTMLIR_TOOLCHAIN_DIR/bin" $PATH

if set -q PYTHONHOME
    set -gx _OLD_VIRTUAL_PYTHONHOME $PYTHONHOME
    set -e PYTHONHOME
end

if test -z "$VIRTUAL_ENV_DISABLE_PROMPT"
    functions -c fish_prompt _old_fish_prompt
    function fish_prompt
        set -l old_status $status
        printf "%s%s%s" (set_color 4B8BBE) "(tt-forge) " (set_color normal)
        _old_fish_prompt
    end
    set -gx _OLD_FISH_PROMPT_OVERRIDE "$VIRTUAL_ENV"
    set -gx VIRTUAL_ENV_PROMPT "(tt-forge) "
end
