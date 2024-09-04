add_custom_target(install_ttforge ALL
    COMMAND cd "${CMAKE_SOURCE_DIR}/forge" && pip install -e .
    COMMENT "Installing ttforge module")

add_dependencies(install_ttforge run_after_ttforge_csrc)
