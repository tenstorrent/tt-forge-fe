# Every variable in subdir must be prefixed with subdir (emulating a namespace)

PYBUDA_CSRC_BACKENDAPI_LIB = $(LIBDIR)/libbackend_api.a
PYBUDA_CSRC_BACKENDAPI_SRCS += \
	pybuda/csrc/backend_api/backend_api.cpp \
	pybuda/csrc/backend_api/arch_type.cpp

PYBUDA_CSRC_BACKENDAPI_INCLUDES = $(PYBUDA_CSRC_INCLUDES) $(BACKEND_INCLUDES)

PYBUDA_CSRC_BACKENDAPI_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BACKENDAPI_SRCS:.cpp=.o))
PYBUDA_CSRC_BACKENDAPI_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_BACKENDAPI_SRCS:.cpp=.d))

-include $(PYBUDA_CSRC_BACKENDAPI_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
pybuda/csrc/backend_api: $(PYBUDA_CSRC_BACKENDAPI_LIB) $(PYBUDA_CSRC_SHARED_UTILS_LIB) ;

$(PYBUDA_CSRC_BACKENDAPI_LIB): $(PYBUDA_CSRC_BACKENDAPI_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

$(OBJDIR)/pybuda/csrc/backend_api/%.o: pybuda/csrc/backend_api/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(PYBUDA_CSRC_BACKENDAPI_INCLUDES) -c -o $@ $<

