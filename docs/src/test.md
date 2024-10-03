# Testing

This page describes how to run different kinds of tests in the `tt-forge-fe` project. If you haven't built the project yet,
please refer to the [Build](build.md) page.

## Unit tests

To build the unit tests, run the following command:

```sh
cmake --build build -- build_unit_tests
```

To run the unit tests (this will also build the tests if they are not built):

```sh
cmake --build build -- run_unit_tests
```

> **Note:** The unit tests are built in the `build/forge/csrc/test` directory. From there, you can run targeted tests directly.
> - For example, to run all the tests defined in `forge/csrc/test/passes/` use: `./build/forge/csrc/test/test_passes`
> - You can further filter the tests by using the `--gtest_filter` flag:
>   ```sh
>   ./build/forge/csrc/test/test_passes --gtest_filter=MMFuseBias/MMFuseBias.mm_fuse_bias/3
>   ```

## End to end tests

For running the end-to-end tests we use the `pytest` framework. To run these tests, you need to be on a machine with a Tenstorrent
Wormhole device. Also, we are still in the process of cleaning up the old tests, so not all tests are working. For a list of green
tests, consult `pytest.ini`.

> **Note:** Make sure that you have activated the python environment before running the tests.

To run all tests defined in `/test/mlir/test_ops.py` use:

```sh
pytest -svv forge/test/mlir/test_ops.py
```

To run a specific test, use the following:

```sh
pytest -svv forge/test/mlir/test_ops.py::test_add
```

> - The `-svv` flag is optional and used to display more information about the test run.
