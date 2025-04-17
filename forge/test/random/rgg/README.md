## RGG Random Graph Generator

Random Graph Generator supports testing of randomly generated graphs. Tests based on RGG can be triggered as regular pytests and executed in a single run of pytest. Tests are performed as a bulk operation for the specified number of tests.

### Execution

For each random model RGG passes through steps:

 - Generate random model for specific random_seed
 - Verify model via verify_module

Source code of each randomly generated model with a pytest function can be automatically stored in a folder, ex `test/random_tests/` if configured.

## Run

Entrypoint for RGG pytests is in `test_graphs.py` module

Example command for running Forge RGG tests generated via random graph algorithm

```shell
LOGURU_LEVEL=DEBUG FRAMEWORKS=FORGE RANDOM_TEST_SEED=0 RANDOM_TEST_COUNT=5 VERIFICATION_TIMEOUT=60 MIN_DIM=3 MAX_DIM=4 MIN_OP_SIZE_PER_DIM=16 MAX_OP_SIZE_PER_DIM=64 OP_SIZE_QUANTIZATION=1 MIN_MICROBATCH_SIZE=1 MAX_MICROBATCH_SIZE=8 NUM_OF_NODES_MIN=5 NUM_OF_NODES_MAX=10 NUM_OF_FORK_JOINS_MAX=5 CONSTANT_INPUT_RATE=20 SAME_INPUTS_PERCENT_LIMIT=10 pytest -svv forge/test/random/test_graphs.py
```

Example command for running PyTorch RGG tests generated via random graph algorithm

```shell
LOGURU_LEVEL=DEBUG FRAMEWORKS=PYTORCH RANDOM_TEST_SEED=0 RANDOM_TEST_COUNT=5 VERIFICATION_TIMEOUT=60 MIN_DIM=4 MAX_DIM=4 MIN_OP_SIZE_PER_DIM=4  MAX_OP_SIZE_PER_DIM=8  OP_SIZE_QUANTIZATION=1 MIN_MICROBATCH_SIZE=1 MAX_MICROBATCH_SIZE=1 NUM_OF_NODES_MIN=3 NUM_OF_NODES_MAX=5  NUM_OF_FORK_JOINS_MAX=5 CONSTANT_INPUT_RATE=20 SAME_INPUTS_PERCENT_LIMIT=10 pytest -svv forge/test/random/test_graphs.py
```


## Helper scripts

To start interacting with RGG tests load helper commands via

```sh
source forge/test/random/test_commands.sh
```

Available commands

| Command               | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `print_help`          | Print commands and current query parameters.                          |
| `print_query_docs`    | Print docs for all available query parameters.                        |
| `print_params`        | Print current query parameters values.                                |
| `select_test_graphs`  | Select test_graps pytest function.                                    |
| `pytest`              | Run all tests specfied via a query parameters.                        |
| `with-params pytest`  | Print params before and after test run.                               |

Usage examples

Run 5 tests for PYTHORCH framework

```sh
RANDOM_TEST_COUNT=5 FRAMEWORKS=PYTORCH with-params pytest
```


## Configuration

Configuration of RGG is supported via `RandomizerConfig`

Parameters includes configuration of:

 - framework
 - number of tests
 - number of nodes
 - min and max size of an operand dimension
 - ...

For more details about configuration please take a look at `forge/test/random/rgg/config.py`.

Please refer to full list of supported enviroment variables in [README.debug.md](../README.debug.md)

*Test specific environment variables that can be used to fine tune default behavior of Forge RGG tests.*

## Parameters

Full list of supported query parameters

| Parameter                     | Description                                                       | Supported values              | Default               |
| ----------------------------- | ----------------------------------------------------------------- | ----------------------------- | --------------------- |
| FRAMEWORKS                    | List of frameworks.                                               | PYTORCH, FORGE                |                       |
| ALGORITHMS                    | List of algorithms.                                               | RANDOM                        | RANDOM                |
| CONFIGS                       | List of config names.                                             | DEFAULT, UNSTABLE, ...        | DEFAULT               |
| RANDOM\_TEST\_SEED            | Initial seed for RGG.                                             |                               | 0                     |
| RANDOM\_TEST\_COUNT           | Number of random tests to be generated and executed. The parameter generate test_index in range from 0 to RANDOM\_TEST\_COUNT-1. |                               | 5                     |
| RANDOM\_TESTS\_SELECTED       | Limiting random tests to only selected subset defined as comma separated list of test indexes. E.x. "3,4,6"    |                               | Default is no limitation if not specified or empty. |
| VERIFICATION\_TIMEOUT         | Limit time for inference verification in seconds.                 |                               | 60                    |
| MIN\_DIM                      | Minimal number of dimensions of input tensors.                    |                               | 3                     |
| MAX\_DIM                      | Maximum number of dimensions of input tensors.                    |                               | 4                     |
| MIN\_OP\_SIZE\_PER\_DIM       | Minimal size of an operand dimension.                             |                               | 16                    |
| MAX\_OP\_SIZE\_PER\_DIM       | Maximum size of an operand dimension. Smaller operand size results in fewer failed tests. |                               | 512                   |
| OP\_SIZE\_QUANTIZATION        | Quantization factor for operand size.                             |                               | 1                     |
| MIN_MICROBATCH_SIZE           | Minimal size of microbatch of an input tensor.                    |                               | 1                     |
| MAX_MICROBATCH_SIZE           | Maximum size of microbatch of an input tensor.                    |                               | 8                     |
| NUM\_OF\_NODES\_MIN           | Minimal number of nodes to be generated by RGG.                   |                               | 5                     |
| NUM\_OF\_NODES\_MAX           | Maximum number of nodes to be generated by RGG.                   |                               | 10                    |
| NUM\_OF\_FORK\_JOINS\_MAX     | Maximum number of fork joins to be generated by random graph algorithm in RGG.    |                               | 50                     |
| CONSTANT\_INPUT\_RATE         | Rate of constant inputs in RGG in percents.                       |                               | 50                    |
| SAME\_INPUTS\_PERCENT\_LIMIT  | Percent limit of nodes which have same value on multiple inputes. |                               | 10                    |


Test configuration parameters

| Parameter                 | Description                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| SKIP_FORGE_VERIFICATION   | Skip Forge model verification including model compiling and inference                         |


## Development

Entrypoint for RGG impplementation is `process_test` module

Parameters of process_test pytest:

 - test_index - index of a test
 - random_seed - random seed of a test
 - test_device - target test device
 - randomizer_config - test configation parameters
 - graph_builder_type - algorithm
 - framework - target framework
