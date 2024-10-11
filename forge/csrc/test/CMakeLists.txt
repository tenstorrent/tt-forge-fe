find_package(GTest REQUIRED)
enable_testing()

# This will hold list of all executable test names.
set(TEST_NAMES)

# Picks up all the files in a given test dir and creates
# a test executable for the given directory.
function (add_unittest TEST_DIR)
    file(GLOB TEST_FILES "${TEST_DIR}/*.cpp")
    set(TEST_NAME "test_${TEST_DIR}")

    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_FILES})
    target_compile_options(${TEST_NAME} PRIVATE
        ${TTFORGE_CSRC_CFLAGS}
        ${CXXFLAGS}
    )
    target_link_libraries(${TEST_NAME} PRIVATE GTest::GTest GTest::Main ttforge_csrc ${TTFORGE_PYTHON_VERSION})

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    list(APPEND TEST_NAMES ${TEST_NAME})
    set(TEST_NAMES ${TEST_NAMES} PARENT_SCOPE)
endfunction()

# Add the unit tests (directories)
add_unittest("passes")
add_unittest("graph_lib")

add_custom_target(build_unit_tests
    COMMENT "Building unit tests..."
    DEPENDS
        ttforge_csrc
        ${TEST_NAMES}
)

add_custom_target(run_unit_tests
    COMMENT "Running unit tests..."
    COMMAND
        ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS
        build_unit_tests
)
