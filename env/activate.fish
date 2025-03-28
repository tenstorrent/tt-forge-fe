function deactivate
    if set -q _OLD_VIRTUAL_PATH
        set -gx PATH $_OLD_VIRTUAL_PATH
        set -e _OLD_VIRTUAL_PATH
    end

    if set -q _OLD_VIRTUAL_PYTHONHOME
        set -gx PYTHONHOME $_OLD_VIRTUAL_PYTHONHOME
        set -e _OLD_VIRTUAL_PYTHONHOME
    end

    if set -q _OLD_FISH_PROMPT_OVERRIDE
        functions -e fish_prompt
        functions -c _old_fish_prompt fish_prompt
        functions -e _old_fish_prompt
        set -e _OLD_FISH_PROMPT_OVERRIDE
    end

    set -e VIRTUAL_ENV
    set -e VIRTUAL_ENV_PROMPT
    set -e TTFORGE_VENV_DIR
    set -e TTMLIR_ENV_ACTIVATED

    functions -e deactivate
end

deactivate nondestructive

set -gx TTFORGE_TOOLCHAIN_DIR $TTFORGE_TOOLCHAIN_DIR "/opt/ttforge-toolchain"[1]
set -gx TTFORGE_VENV_DIR $TTFORGE_VENV_DIR "$TTFORGE_TOOLCHAIN_DIR/venv"[1]
set -gx TTMLIR_TOOLCHAIN_DIR $TTMLIR_TOOLCHAIN_DIR "/opt/ttmlir-toolchain"[1]
set -gx TTMLIR_VENV_DIR $TTMLIR_VENV_DIR "$TTMLIR_TOOLCHAIN_DIR/venv"[1]
set -gx TTMLIR_ENV_ACTIVATED 1
set -gx ARCH_NAME $ARCH_NAME "wormhole_b0"[1]
set -gx VIRTUAL_ENV "$TTFORGE_VENV_DIR"

set -gx _OLD_VIRTUAL_PATH $PATH
set -gx PATH "$TTFORGE_VENV_DIR/bin" "$TTMLIR_TOOLCHAIN_DIR/bin" $PATH

if set -q PYTHONHOME
    set -gx _OLD_VIRTUAL_PYTHONHOME $PYTHONHOME
    set -e PYTHONHOME
end

functions -c fish_prompt _old_fish_prompt
function fish_prompt
    printf "(tt-forge) "
    _old_fish_prompt
end
set -gx _OLD_FISH_PROMPT_OVERRIDE 1
set -gx VIRTUAL_ENV_PROMPT "(tt-forge) "
