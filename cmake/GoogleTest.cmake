# GoogleTest configuration for TTForge
# This module handles fetching and setting up GoogleTest v1.17.0

if(TTFORGE_UNITTESTS_ENABLED)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
    )

    # Prevent GoogleTest from being installed with our project
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    message(STATUS "GoogleTest v1.17.0 fetched and configured")
endif()
