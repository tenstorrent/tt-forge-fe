# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYBUDA_CSRC_INCLUDES = \
	-Ipybuda/csrc \
	-Ithird_party/json \
	-I/usr/include/$(PYTHON_VERSION) \
	-isystem $(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/torch/include \
 	-isystem $(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/torch/include/torch/csrc/api/include \
	-I/opt/ttmlir-toolchain/include \
	-Ithird_party/tt-mlir/build/include \
	-Ithird_party/tt-mlir/runtime/include \
	-Ithird_party/tt-mlir/include

PYBUDA_CSRC_WARNINGS ?= -Wall -Wextra -Wno-pragmas -Wno-unused-parameter
PYBUDA_CSRC_CFLAGS ?= $(CFLAGS_NO_WARN) $(PYBUDA_CSRC_WARNINGS) -DUTILS_LOGGER_PYTHON_OSTREAM_REDIRECT=1
TORCH_LIB_DIR = $(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/torch/lib

PYBUDA_CSRC_LIB = $(LIBDIR)/libpybuda_csrc.so
TTMLIR_TOOLCHAIN_DIR = /opt/ttmlir-toolchain
RUNTIME_LIB_DIR = third_party/tt-mlir/third_party/tt-metal/src/tt-metal-build/lib
MLIR_LIB_DIR = -L$(TTMLIR_TOOLCHAIN_DIR)/lib -Lthird_party/tt-mlir/build/lib -Lthird_party/tt-mlir/build/runtime/lib
MLIR_LIBS = -Wl,-rpath,$(TTMLIR_TOOLCHAIN_DIR)/lib -lLLVM -lMLIR
TT_MLIR_LIBS = -lMLIRTTDialect -lMLIRTTIRDialect -lMLIRTTNNDialect -lMLIRTTIRTransforms -lMLIRTTNNTransforms -lMLIRTTKernelDialect -lMLIRTTMetalDialect -lMLIRTTIRAnalysis
RUNTIME_LIBS = -lTTRuntime -lTTRuntimeTTNN -L$(RUNTIME_LIB_DIR) -Wl,-rpath,\$$ORIGIN/../../$(RUNTIME_LIB_DIR) -l:_ttnn.so -ltt_metal -ldevice -ltt_eager

include pybuda/csrc/graph_lib/module.mk
include pybuda/csrc/shared_utils/module.mk
include pybuda/csrc/autograd/module.mk
include pybuda/csrc/reportify/module.mk
include pybuda/csrc/backend_api/module.mk
include pybuda/csrc/tt_torch_device/module.mk
include pybuda/csrc/runtime/module.mk

PYBUDA_CSRC_LDFLAGS = -Wl,-rpath,\$$ORIGIN/../python_env/lib/$(PYTHON_VERSION)/site-packages/torch/lib -ltorch -ltorch_cpu -lc10 -ltorch_python $(PYTHON_LDFLAGS) -l$(PYTHON_VERSION) $(MLIR_LIB_DIR) $(MLIR_LIBS) $(TT_MLIR_LIBS) $(RUNTIME_LIBS) -lm -lz -lcurses -lxml2 -lflatbuffers

PYBUDA_CSRC_SRCS = \
		pybuda/csrc/pybuda_bindings.cpp \
		pybuda/csrc/buda_passes.cpp \
		$(wildcard pybuda/csrc/passes/*.cpp) \
		pybuda/csrc/lower_to_buda/common.cpp

PYBUDA_CSRC_OBJS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_SRCS:.cpp=.o))
PYBUDA_CSRC_DEPS = $(addprefix $(OBJDIR)/, $(PYBUDA_CSRC_SRCS:.cpp=.d))

PYBUDA_THIRD_PARTY_DEPS = $(SUBMODULESDIR)/third_party/pybind11.checkout

-include $(PYBUDA_CSRC_DEPS)

$(PYBUDA_CSRC_LIB): $(PYBUDA_CSRC_OBJS) $(PYBUDA_CSRC_GRAPH_LIB) $(PYBUDA_CSRC_AUTOGRAD) $(PYBUDA_CSRC_PATTERN_MATCHER_LIB) $(PYBUDA_CSRC_BALANCER_LIB) $(PYBUDA_CSRC_PLACER_LIB) $(PYBUDA_CSRC_SCHEDULER_LIB) $(PYBUDA_CSRC_REPORTIFY) $(PYBUDA_CSRC_BACKENDAPI_LIB) $(PYBUDA_CSRC_SHARED_UTILS_LIB) $(PYBUDA_CSRC_PERF_MODEL_LIB) $(PYBUDA_CSRC_TT_TORCH_DEVICE_LIB) $(PYBUDA_CSRC_RUNTIME_LIB)
	@mkdir -p $(LIBDIR)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) -L$(TORCH_LIB_DIR) -o $@ $^ $(LDFLAGS) $(PYBUDA_CSRC_LDFLAGS)

$(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/pybuda/_C.so: $(PYBUDA_CSRC_LIB)
	@mkdir -p $(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/pybuda
	cp $^ $@
	touch -r $^ $@
	ln -sf ../../$(PYTHON_ENV_ROOT)/lib/$(PYTHON_VERSION)/site-packages/pybuda/_C.so pybuda/pybuda/_C.so

$(OBJDIR)/pybuda/csrc/%.o: pybuda/csrc/%.cpp $(PYTHON_ENV) $(PYBUDA_THIRD_PARTY_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(PYBUDA_CSRC_CFLAGS) $(CXXFLAGS) $(SHARED_LIB_FLAGS) $(PYBUDA_CSRC_INCLUDES) -c -o $@ $<

pybuda/csrc: $(PYBUDA_CSRC_LIB) ;
