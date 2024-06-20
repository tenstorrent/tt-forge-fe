.SUFFIXES:

OS ?= $(shell uname)

ifeq ($(OS),Linux)
MAKEFLAGS := --jobs=$(shell nproc) --output-sync=target
else ifeq ($(OS),Darwin)
MAKEFLAGS := --jobs=$(shell sysctl -n hw.physicalcpu) --output-sync=target
else
$(error "Unknown OS: $(OS)")
endif

# Setup CONFIG, DEVICE_RUNNER, and out/build dirs first
CONFIG ?= assert
ARCH_NAME ?= grayskull
# TODO: enable OUT to be per config (this impacts all scripts that run tests)
# OUT ?= build_$(DEVICE_RUNNER)_$(CONFIG)
OUT ?= build
PREFIX ?= $(OUT)

CONFIG_CFLAGS =
CONFIG_LDFLAGS =
CONFIG_CXXFLAGS =

ifeq ($(CONFIG), release)
CONFIG_CFLAGS += -O3
CONFIG_CXXFLAGS = -fvisibility-inlines-hidden
else ifeq ($(CONFIG), ci)  # significantly smaller artifacts
CONFIG_CFLAGS += -O3 -DDEBUG -Werror
else ifeq ($(CONFIG), assert)
CONFIG_CFLAGS += -O3 -g -DDEBUG
else ifeq ($(CONFIG), asan)
CONFIG_CFLAGS += -O3 -g -DDEBUG -fsanitize=address
CONFIG_LDFLAGS += -fsanitize=address
ifeq ($(findstring clang,$(CC)),clang)
CONFIG_CFLAGS += -shared-libasan
CONFIG_LDFLAGS += -shared-libasan
endif
else ifeq ($(CONFIG), ubsan)
CONFIG_CFLAGS += -O3 -g -DDEBUG -fsanitize=undefined
CONFIG_LDFLAGS += -fsanitize=undefined
else ifeq ($(CONFIG), gprof)
CONFIG_CFLAGS += -O3 -g -DDEBUG -pg
CONFIG_LDFLAGS += -pg
else ifeq ($(CONFIG), debug)
CONFIG_CFLAGS += -O0 -g -DDEBUG
else
$(error Unknown value for CONFIG "$(CONFIG)")
endif

OBJDIR = $(OUT)/obj
LIBDIR = $(OUT)/lib
BINDIR = $(OUT)/bin
INCDIR = $(OUT)/include
TESTDIR = $(OUT)/test
DOCSDIR = $(OUT)/docs
SUBMODULESDIR = $(OUT)/submodules

# Python version
PYTHON_VERSION ?= python3.11
PYTHON_INCLUDES = $(shell $(PYTHON_VERSION)-config --includes)
PYTHON_LDFLAGS = $(shell $(PYTHON_VERSION)-config --ldflags)

# Top level flags, compiler, defines etc.

#WARNINGS ?= -Wall -Wextra
WARNINGS ?= -Wdelete-non-virtual-dtor -Wreturn-type -Wswitch -Wuninitialized -Wno-unused-parameter
CC ?= clang
CXX ?= clang++
CFLAGS_NO_WARN ?= -MMD -I. $(CONFIG_CFLAGS) -DBUILD_DIR=\"$(OUT)\" -I$(INCDIR) -DFMT_HEADER_ONLY -Ithird_party/fmt -Ithird_party/pybind11/include $(PYTHON_INCLUDES)
CFLAGS ?= $(CFLAGS_NO_WARN) $(WARNINGS)
CXXFLAGS ?= --std=c++17 $(CONFIG_CXXFLAGS)
LDFLAGS ?= $(CONFIG_LDFLAGS) -Wl,-rpath,$(PREFIX)/lib -L$(LIBDIR) -v
SHARED_LIB_FLAGS = -shared -fPIC
STATIC_LIB_FLAGS = -fPIC
ifeq ($(findstring clang,$(CC)),clang)
WARNINGS += -Wno-c++11-narrowing
LDFLAGS += -lstdc++
else
WARNINGS += -Wmaybe-uninitialized
LDFLAGS += -lstdc++
endif
GIT_COMMON_DIR=$(shell git rev-parse --git-common-dir)
SUBMODULES=$(shell git submodule status | grep -o "third_party/[^ ]*")
SUBMODULES_UPDATED=$(addprefix $(SUBMODULESDIR)/, $(SUBMODULES:%=%.checkout))
SKIP_BBE_UPDATE ?= 0
SKIP_SUBMODULE_UPDATE ?= $(SKIP_BBE_UPDATE)

ifeq ($(EMULATION_DEVICE_EN), 1)
    TENSIX_EMULATION_ZEBU = $(TENSIX_EMULATION_ROOT)/zebu
    TENSIX_EMULATION_ZCUI_WORK = $(TENSIX_EMULATION_ROOT)/targets/tensix_2x2_1dram_BH/zcui.work
endif

all: update_submodules build ;

# These must be in dependency order (enforces no circular deps)
include python_env/module.mk
include pybuda/module.mk
include docs/public/module.mk

update_submodules: $(SUBMODULES_UPDATED) emulation_device_links ;

$(SUBMODULESDIR)/%.checkout:
	@mkdir -p $(dir $@)
ifeq ($(SKIP_SUBMODULE_UPDATE), 0)
	git submodule update --init --recursive $(@:$(SUBMODULESDIR)/%.checkout=%)
	#git -C $(@:$(SUBMODULESDIR)/%.checkout=%) submodule foreach --recursive git lfs install || true
	#git -C $(@:$(SUBMODULESDIR)/%.checkout=%) submodule foreach --recursive git lfs pull
	#git -C $(@:$(SUBMODULESDIR)/%.checkout=%) submodule foreach --recursive git lfs checkout HEAD
endif
	touch $@

emulation_device_links:
ifeq ($(EMULATION_DEVICE_EN), 1)
	@echo "Linking and copying emulation device files..."
	ln -sf $(TENSIX_EMULATION_ZCUI_WORK) $(OUT)/.
	ln -sf $(TENSIX_EMULATION_ZCUI_WORK) ../
	cp -f $(TENSIX_EMULATION_ZEBU)/scripts/designFeatures ./
endif

build: pybuda third_party/tvm ;

third_party/tvm: $(SUBMODULESDIR)/third_party/tvm.build ;

$(SUBMODULESDIR)/third_party/tvm.build: $(PYTHON_ENV) $(SUBMODULESDIR)/third_party/tvm.checkout
	bash -c "source $(PYTHON_ENV_ROOT)/bin/activate && ./third_party/tvm/install.sh"
	touch $@

build_tests: pybuda/csrc/graph_lib/tests pybuda/csrc/passes/tests ;

run_tests: build_tests
	@echo "Running tests..."
	$(TESTDIR)/pybuda/csrc/graph_lib/tests/graphlib_unit_tests
	$(TESTDIR)/pybuda/csrc/passes/tests/passes_unit_tests

clean: 
	rm -rf $(OUT)
	rm -rf third_party/tvm/build
	rm -rf build_deps/

clean_no_python:
	find $(OUT)/ -maxdepth 1 -mindepth 1 -type d -not -name 'python_env' -print0 | xargs -0 -I {} rm -Rf {}
	find $(OUT)/ -maxdepth 1 -type f -delete


.PHONY: install
install: all
ifeq ($(PREFIX), $(OUT))
	@echo "To install you must set PREFIX, e.g."
	@echo ""
	@echo "  PREFIX=/usr CONFIG=release make install"
	@echo ""
	@exit 1
endif
	cp -r $(LIBDIR)/* $(PREFIX)/lib/
	cp -r $(INCDIR)/* $(PREFIX)/include/
	cp -r $(BINDIR)/* $(PREFIX)/bin/

.PHONY: b0
b0: export LD_LIBRARY_PATH=versim/wormhole_b0/lib:versim/wormhole_b0/lib/ext
b0: export BACKEND_ARCH_NAME=wormhole_b0
b0: build ;

.PHONY: build_tvm
build_tvm: third_party/tvm ;

.PHONY: stubs
stubs:
	pip install mypy
	stubgen -m pybuda._C -m pybuda._C.autograd -m pybuda._C.graph -m pybuda._C.torch_device -o pybuda

# Cleaning PyBuda and BBE artifacts
.PHONY: clean_tt
clean_tt:
	@rm -rf .hlkc_cache/ .pkl_memoize_py3/ generated_modules/ tt_build/
	@rm -f *netlist.yaml
