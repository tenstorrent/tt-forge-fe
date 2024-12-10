# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
from typing import List, Optional
from dataclasses import dataclass
import subprocess

import pytest
import _pytest.skipping
import torch.multiprocessing as mp
import torch
import tensorflow as tf

# This is a workaround to set RTLD_GLOBAL flag to load emulation ZeBu library.
# Essentially symbol names have to be unique in global scope to work with ZeBu,
# hence need to be set as GLOBAL. This is a requirement for ZeBu.
import sys

original_flags = sys.getdlopenflags()
if os.environ.get("FORGE_ENABLE_EMULATION_DEVICE") == "1":
    sys.setdlopenflags(os.RTLD_LAZY | os.RTLD_GLOBAL)
# Import code that requires os.RTLD_GLOBAL goes here
# Reset the flags to their original value
if os.environ.get("FORGE_ENABLE_EMULATION_DEVICE") == "1":
    sys.setdlopenflags(original_flags)

import forge
from forge.verify.config import TestKind
from forge.torch_compile import reset_state

from forge.config import _get_global_compiler_config

collect_ignore = ["legacy_tests"]


def pytest_sessionstart(session):
    # See: https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
    mp.set_start_method("spawn")
    num_threads = 8
    if "FORGE_NUM_THREADS" in os.environ:
        num_threads = int(os.environ["FORGE_NUM_THREADS"])
    torch.set_num_threads(num_threads)
    mp.set_sharing_strategy("file_system")
    os.environ["TVM_NUM_THREADS"] = f"{num_threads}"
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    torch._dynamo.reset()
    reset_state()
    # If specified by env variable, print the environment variables
    # It can be useful in CI jobs to get the state of the enviroment variables before test session starts
    print_env_variables = bool(int(os.environ.get("PYTEST_PRINT_ENV_VARIABLES", "0")))
    if print_env_variables:
        forge_specific_vars = {}
        tt_backend_specific_vars = {}
        print(f"####### Environment variables - Count: {len(os.environ)} #######")
        for key, value in os.environ.items():
            print(f"{key}={value}")
            if key.startswith("FORGE_") or key.startswith("GOLDEN_"):
                forge_specific_vars[key] = value
            elif key.startswith("TT_BACKEND_"):
                tt_backend_specific_vars[key] = value

        print(f"####### FORGE specific enviroment variables - Count: {len(forge_specific_vars)} #######")
        for key, value in forge_specific_vars.items():
            print(f"{key}={value}")

        print(f"####### TT_BACKEND specific enviroment variables - Count: {len(tt_backend_specific_vars)} #######")
        for key, value in tt_backend_specific_vars.items():
            print(f"{key}={value}")


@pytest.fixture(autouse=True)
def clear_forge():
    if "FORGE_RESET_DEV_BEFORE_TEST" in os.environ:
        # Reset device between tests
        # For this to work, pytest must be called with --forked
        subprocess.check_call(["device/bin/silicon/reset.sh"], cwd=os.environ["FORGE_HOME"])

    import random

    random.seed(0)

    import numpy as np

    np.random.seed(0)

    torch.manual_seed(0)

    import tensorflow as tf

    tf.random.set_seed(0)

    yield

    # clean up after each test
    forge.forge_reset()
    torch._dynamo.reset()
    reset_state()


def pytest_addoption(parser):
    parser.addoption(
        "--generate-unique-op-tests",
        action="store_true",
        default=False,
        help="Generate unique op tests for the given model",
    )
    parser.addoption(
        "--silicon-only", action="store_true", default=False, help="run silicon tests only, skip golden/model"
    )
    parser.addoption("--no-silicon", action="store_true", default=False, help="skip silicon tests")
    parser.addoption(
        "--compile-only", action="store_true", default=False, help="only compiles the model and generate TTI"
    )
    parser.addoption(
        "--run-only", action="store_true", default=False, help="load the generated TTI and only runs the model"
    )
    parser.addoption(
        "--tti-path",
        default=None,
        type=str,
        help="Valid only if either --compile-only or --run-only is specified. Save/load TTI from the path",
    )
    parser.addoption(
        "--device-config", default=None, type=str, help="Runtime yaml is automatically configured based on the value"
    )
    parser.addoption(
        "--devtype",
        default=None,
        type=str,
        choices=("golden", "silicon"),
        help="Valid only if --compile-only is specified. Set the backend device type between Golden or Silicon",
    )
    parser.addoption(
        "--no-skips", action="store_true", default=False, help="ignore pytest.skip() calls, and continue on with test"
    )


"""
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--silicon", action="store_true", default=False, help="run silicon tests"
    )
    parser.addoption(
        "--versim", action="store_true", default=False, help="run versim tests"
    )
    parser.addoption(
        "--arch", action="store", default="grayskull", help="run tests on different arch"
    )
    parser.addoption(
        "--microbatch-size", action="store", default=8,
    )
    parser.addoption(
        "--num-microbatches", action="store", default=2,
    )
    parser.addoption(
        "--batch-size", action="store", default=32,
    )
    parser.addoption(
        "--num-batches", action="store", default=4,
    )
    parser.addoption(
        "--num-chips", action="store", default=1,
    )

# Use this to stop slow tests from running by default
@pytest.fixture(name="runslow", scope="session", autouse=True)
def runslow(request):
   return request.config.getoption("--runslow")
"""


@pytest.fixture(autouse=True, scope="session")
def initialize_global_compiler_configuration_based_on_pytest_args(pytestconfig):
    """
    Set the global compiler config options for the test session
    which will generate op tests for the given model.
    """
    compiler_cfg = _get_global_compiler_config()

    compiler_cfg.tvm_generate_unique_op_tests = pytestconfig.getoption("--generate-unique-op-tests")

    if compiler_cfg.tvm_generate_unique_op_tests:
        # For running standalone tests, we need to retain the generated python files
        # together with stored model parameters
        compiler_cfg.retain_tvm_python_files = True

        # Required to prevent early tensor deallocation
        compiler_cfg.enable_op_level_comparision = True


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_preparse(config, args):

    if "--no-skips" not in args:
        return

    def no_skip(*args, **kwargs):
        return

    pytest.skip = no_skip
    _pytest.skipping.skip = no_skip  # can't run skipped tests with decorator @pytest.mark.skip without this


# DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE = {
#     "gs_e150": BackendDevice.Grayskull,
#     "gs_e300": BackendDevice.Grayskull,
#     "wh_n150": BackendDevice.Wormhole_B0,
#     "wh_n300": BackendDevice.Wormhole_B0,
#     "galaxy": BackendDevice.Wormhole_B0,
# }

# @dataclass
# class TestDevice:
#     devtype: BackendType
#     arch: BackendDevice
#     devmode: DeviceMode
#     tti_path: str = None

#     @classmethod
#     def from_str(cls, name: str, devmode: DeviceMode, tti_path: str = None, device_config=None) -> "TestDevice":
#         if name == "Golden":
#             if device_config and DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE.get(device_config, None):
#                 return TestDevice(devtype=BackendType.Golden, arch=DEVICE_CONFIG_TO_BACKEND_DEVICE_TYPE[device_config], devmode=devmode, tti_path=tti_path)
#             elif "GOLDEN_WORMHOLE_B0" in os.environ:
#                 return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Wormhole_B0, devmode=devmode, tti_path=tti_path)
#             elif "FORGE_GOLDEN_BLACKHOLE" in os.environ:
#                 return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Blackhole, devmode=devmode, tti_path=tti_path)
#             return TestDevice(devtype=BackendType.Golden, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
#         if name == "Model":
#             return TestDevice(devtype=BackendType.Model, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
#         if name == "Versim":
#             # Set default versim device arch to Grayskull
#             versim_backend_device = BackendDevice.Grayskull
#             # If FORGE_VERSIM_DEVICE_ARCH is set, use that arch for Versim device
#             versim_arch_name = os.environ.get("FORGE_VERSIM_DEVICE_ARCH", None)
#             if versim_arch_name != None:
#                 versim_backend_device = BackendDevice.from_string(versim_arch_name)
#             return TestDevice(devtype=BackendType.Versim, arch=versim_backend_device, devmode=devmode, tti_path=tti_path)
#         if name == "Emulation":
#             # Set default emulation device arch to Grayskull
#             emulation_backend_device = BackendDevice.Grayskull
#             # If FORGE_EMULATION_DEVICE_ARCH is set, use that arch for Emulation device
#             emulation_arch_name = os.environ.get("FORGE_EMULATION_DEVICE_ARCH", None)
#             if emulation_arch_name != None:
#                 emulation_backend_device = BackendDevice.from_string(emulation_arch_name)
#             return TestDevice(devtype=BackendType.Emulation, arch=emulation_backend_device, devmode=devmode, tti_path=tti_path)
#         if name == "Grayskull":
#             return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Grayskull, devmode=devmode, tti_path=tti_path)
#         if name == "Wormhole_B0":
#             return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Wormhole_B0, devmode=devmode, tti_path=tti_path)
#         if name == "Blackhole":
#             return TestDevice(devtype=BackendType.Silicon, arch=BackendDevice.Blackhole, devmode=devmode, tti_path=tti_path)
#         raise RuntimeError("Unknown test device: " + name)

#     def is_available(self, device_list: List[BackendDevice], silicon_only: bool, no_silicon: bool, devtype: Optional[BackendType], devmode: DeviceMode) -> bool:
#         """
#         Return true if this kind of device is available on the current host. Expect a list of devices from
#         `detect_available_devices`.
#         """
#         if devtype is not None and self.devtype != devtype:
#             return False

#         if self.devtype == BackendType.Golden:
#             return not silicon_only

#         if self.devtype == BackendType.Model:
#             return bool(int(os.environ.get("FORGE_ENABLE_MODEL_DEVICE", "0")))

#         if self.devtype == BackendType.Versim:
#             return bool(int(os.environ.get("FORGE_ENABLE_VERSIM_DEVICE", "0")))

#         if self.devtype == BackendType.Emulation:
#             return bool(int(os.environ.get("FORGE_ENABLE_EMULATION_DEVICE", "0")))

#         if self.devtype == BackendType.Silicon:
#             compiled_arch_name = os.environ.get("BACKEND_ARCH_NAME", None) or os.environ.get("ARCH_NAME", None)
#             if compiled_arch_name == "wormhole_b0":
#                 compiled_arch = BackendDevice.Wormhole_B0
#             elif compiled_arch_name == "blackhole":
#                 compiled_arch = BackendDevice.Blackhole
#             else:
#                 compiled_arch = BackendDevice.Grayskull

#             is_offline_silicon_compile = devmode == DeviceMode.CompileOnly and self.arch == compiled_arch
#             return (self.arch in device_list and not no_silicon) or is_offline_silicon_compile

#         return False

#     def is_silicon(self):
#         return self.devtype == BackendType.Silicon

#     def is_grayskull(self):
#         return self.arch == BackendDevice.Grayskull

#     def is_wormhole_b0(self):
#         return self.arch == BackendDevice.Wormhole_B0

#     def is_blackhole(self):
#         return self.arch == BackendDevice.Blackhole

device_cfg_global = None


def pytest_generate_tests(metafunc):
    global device_cfg_global

    if "test_kind" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_kind",
            (TestKind.INFERENCE, TestKind.TRAINING, TestKind.TRAINING_RECOMPUTE),
            ids=["inference", "training", "training_with_recompute"],
        )

    if "training" in metafunc.fixturenames:
        metafunc.parametrize("training", (False, True), ids=["inference", "training"])

    if "test_device" in metafunc.fixturenames:
        # Temporary work arround to provide dummy test_device
        # TODO remove workarround https://github.com/tenstorrent/tt-forge-fe/issues/342
        metafunc.parametrize("test_device", (None,), ids=["no_device"])

    if "_test_device_not_implemented" in metafunc.fixturenames:
        # if "test_device" in metafunc.fixturenames:
        names = ["Golden", "Model", "Versim", "Emulation", "Grayskull", "Wormhole_B0", "Blackhole"]

        # Set device-mode for the test
        compile_only = metafunc.config.getoption("--compile-only")
        run_only = metafunc.config.getoption("--run-only")
        devtype = metafunc.config.getoption("--devtype")
        devtype = BackendType.from_string(devtype.capitalize()) if devtype else None

        devmode = DeviceMode.CompileAndRun
        if compile_only:
            devmode = DeviceMode.CompileOnly
            if devtype is None:
                assert False, "Backend device type needs to be specified when running tests with compile-only mode"
        elif run_only:
            devmode = DeviceMode.RunOnly

        # Configure TTI-path only if compile/run-only is set
        tti_path = None
        if compile_only or run_only:
            tti_path = metafunc.config.getoption("--tti-path")

        devices = [(TestDevice.from_str(s, devmode, tti_path, device_cfg_global), s) for s in names]
        silicon_only = metafunc.config.getoption("--silicon-only")
        no_silicon = metafunc.config.getoption("--no-silicon")
        device_list = []
        if not no_silicon:
            device_list = detect_available_devices()
        enabled_devices = [
            (d, name)
            for (d, name) in devices
            if d.is_available(device_list, silicon_only, no_silicon, devtype, devmode)
        ]
        params = [pytest.param(d) for (d, _) in enabled_devices]
        ids = [name for (_, name) in enabled_devices]

        metafunc.parametrize("test_device", params, ids=ids)

    # Configure backend runtime yaml
    device_cfg_global = metafunc.config.getoption("--device-config")


environ_before_test = None


def pytest_runtest_logreport(report):
    if report.when == "setup":
        global environ_before_test
        environ_before_test = os.environ.copy()

        global device_cfg_global
        if device_cfg_global:
            forge.set_configuration_options(device_config=device_cfg_global)

        if "FORGE_OVERRIDES_VETO" in os.environ:
            from forge.config import _set_forge_override_veto

            # This functionality represents one way to control general and env based compiler configuration (enable us to
            # add/update/remove existing configs in each test with ease during runtime). In sum, it uses a dict of key-value pairs
            # that control all Forge specific overrides set in test. Have in  mind that this doesn't apply for everything set
            # outside of the test itself (e.g. env vars set before calling the specific pytest).
            #
            # Input to this function is represented as two dicts:
            # - first one is a dict of keys/value pairs that controls general compiler config settings
            # - second one is a dict of keys/value pairs that controls configurations set through environment variables
            #
            # Also, few general notes of how to use this these dicts to control the general and env based compiler configurations:
            # - overriding value with "" will use the value set in test itself
            # - overriding with some specific value will use that config and override it (ignoring the test config)
            # - not including the key and value here, will use default compiler config value (discarding the test config if set there)
            #
            # Description of override levels:
            # - Level 0 - set by compiler;      we want to keep them            (defined during compiler runtime; not in test itself)
            # - Level 1 - set by user in test;  we want to keep them            (defined in test, but easy to use and understandable for end user)
            # - Level 2 - set by dev in test;   we want to remove them          (e.g. enable/disable by default, redefine as more user friendly, etc.)
            # - Level 3 - set by dev in test;   we want to remove them entirely (purely for testing purposes)
            #
            if "FORGE_OVERRIDES_VETO_CUSTOM_SETUP" in os.environ:
                _set_forge_override_veto(
                    {
                        "backend_output_dir": "",
                    },
                    {},
                )
            else:
                _set_forge_override_veto(
                    {
                        "backend_output_dir": "",
                        "backend_runtime_params_path": "",
                        "harvesting_mask": "",
                        "cpu_fallback_ops": "",
                        # Level 1 overrides
                        "balancer_policy": "",
                        "enable_t_streaming": "",
                        "default_df_override": "",
                    },
                    {
                        # Level 2 overrides
                        "FORGE_RIBBON2": "",
                        "FORGE_DISABLE_STREAM_OUTPUT": "",
                        "FORGE_PAD_OUTPUT_BUFFER": "",
                        "FORGE_OVERRIDE_DEVICE_YAML": "",  # Mostly used for 1x1 model overrides
                    },
                )

    elif report.when == "teardown":
        environ_before_test_keys = set(environ_before_test.keys())
        environ_after_test_keys = set(os.environ.keys())

        # remove
        added_flags = environ_before_test_keys ^ environ_after_test_keys
        for f in added_flags:
            os.environ.pop(f, None)

        # reset
        for key, default_value in environ_before_test.items():
            if os.environ.get(key, "") != default_value:
                os.environ[key] = default_value


def pytest_collection_modifyitems(config, items):

    marker = config.getoption("-m")  # Get the marker from the -m option

    if marker and marker == "model_analysis":  # If a marker is specified
        print("Automatic Model Analysis Collected tests: ")
        test_count = 0
        for item in items:
            if marker in item.keywords:
                test_file_path = item.location[0]
                test_name = item.location[2]
                print(f"{test_file_path}::{test_name}")
                test_count += 1
        print(f"Automatic Model Analysis Collected test count: {test_count}")
        if test_count == 0:  # Warn if no tests match the marker
            print(f"Warning: No tests found with marker '{marker}'.")
    else:
        print(items)
