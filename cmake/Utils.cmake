### Utility functions for pybuda ###

### Check if an environment variable exists ###
function(check_env_variable_internal VARIABLE_NAME ret)
    if(NOT DEFINED ENV{${VARIABLE_NAME}})
        set(${ret} "false" PARENT_SCOPE)
    endif()
endfunction()

### Check if an environment variable exists ###
function(check_required_env_var VARIABLE_NAME)
    set(VARIABLE_EXISTS "true")
    check_env_variable_internal(${VARIABLE_NAME} VARIABLE_EXISTS)
    if(NOT ${VARIABLE_EXISTS})
        message(FATAL_ERROR "${VARIABLE_NAME} does not exist. Did you run source env/activate?")
    endif()
endfunction()
