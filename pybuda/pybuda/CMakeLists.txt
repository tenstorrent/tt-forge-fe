add_custom_target(install_pybuda ALL
    COMMAND cd "${CMAKE_SOURCE_DIR}/pybuda" && pip install -e .
    COMMENT "Installing pybuda module")

add_dependencies(install_pybuda run_after_pybuda_csrc)
