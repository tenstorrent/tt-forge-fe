add_library(autograd
    STATIC
    autograd.cpp
    binding.cpp
    python_bindings.cpp)

target_compile_options(autograd PRIVATE ${STATIC_LIB_FLAGS} ${PYBUDA_CSRC_CFLAGS})

