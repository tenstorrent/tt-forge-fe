# Tools

This section will cover setup of various tools that can help you with development of tt-forge-fe.

## Pre-commit

We have defined various pre-commit hooks that check the code for formatting, licensing issues, etc.

To install pre-commit, run the following command:

```sh
source env/activate
pip install pre-commit
```

After installing pre-commit, you can install the hooks by running:

```sh
pre-commit install
```

Now, each time you run `git commit` the pre-commit hooks (checks) will be executed.

If you have already committed before installing the pre-commit hooks, you can run on all files to "catch up":

```sh
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)

## mdbook

We use `mdbook` to generate the documentation. To install `mdbook` on Ubuntu, run the following commands:

```sh
sudo apt install cargo
cargo install mdbook
```

>**NOTE:** If you don't want to install `mdbook` via cargo (Rust package manager), or this doesn't work for you, consult the [official mdbook installation guide](https://rust-lang.github.io/mdBook/cli/index.html).

## Gather Unique Ops Configuration

The model's unique ops configuration can be gathered, and the results can either be printed to the console or saved as a CSV file.

1. **FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT**
   - By setting this flag to one of the following options, the model's unique ops configuration can be extracted at a specific compilation stage or across all stages:

     - **`FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT = ALL`**
       Extracts all the unique ops configurations present in the graph at every compilation stage.

     - **`FORGE_EXTRACT_UNIQUE_OP_CONFIG_AT = {INITIAL_GRAPH / PRE_OPTIMIZE_GRAPH / POST_OPTIMIZE_GRAPH / POST_AUTOGRAD_GRAPH / PRE_LOWERING_GRAPH}`**
       Extracts the unique ops configuration only at the specified compilation stage.

2. **FORGE_PRINT_UNIQUE_OP_CONFIG**
   - By setting this flag to `1`, all unique configurations will be printed to the console.

3. **FORGE_EXPORT_UNIQUE_OP_CONFIG_TO_CSV**
   - By setting this flag to `1`, all unique configurations will be exported to a CSV file. The file can be saved to the default path (i.e., the current directory), or it can be saved to a specific path by setting the `FORGE_EXPORT_UNIQUE_OP_CONFIG_DIR_PATH` environment variable.

> **Note:**
> The delimiter used in the CSV file will be a hyphen (`-`) to avoid potential parsing issues. Commas (`,`) may appear in the op shapes and attributes, which could lead to misinterpretation of the data.

## Cross Correlate Models and Ops

The models and ops can be cross-correlated by running the `scripts/export_models_ops_correlation.py` python script.

The script will perform the following tasks:

1. Run all models until the compile depth specified by the user.
2. Export unique op requirements to a file (each model variants has its own directory, in that directory each compile depth has its own file).
3. Parse those unique op requirements and create a xlsx file that can be loaded into a google sheet.
   1. The xlsx file will contain list of models on X axis (i.e. columns) and list of ops on Y axis (i.e. rows/indices).
   2. Elements in between will contain a checkmark if the desired op from the Y axis (i.e., rows/indices) exists in the model on X axis (i.e., columns).
   3. Models will be sorted alphabetically.
   4. Ops will be sorted by the number of occurrences in the models.

### Usage

To run the script, use the following command:

```sh
python scripts/export_models_ops_correlation.py
```

### Required Options:

|                              **Option**                                   |                                                **Description**                                                   |
| :-----------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| `-c`, `--compile_depth (GENERATE_INITIAL_GRAPH, PRE_LOWERING_PASS, etc.)` | Choose the compilation depth for extracting ops configuration for the models present in `pytest_directory_path`. |
| `-i`, `--pytest_directory_path`                                           | Specify the directory path containing models to test.                                                            |

### Optional Options:

|                              **Option**                                   |                                                **Description**                                                   |
| :-----------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| `-f`, `--output_file_name`                                                | Specify the output file name for the xlsx/csv file.                                                              |
| `-o`, `--output_directory_path`                                           | Specify the output directory path for saving the xlsx/csv file.                                                  |
| `-s`, `--do_save_xlsx`                                                    | Specify whether to save the output in xlsx format.                                                               |

### Example:

```sh
python scripts/export_models_ops_correlation.py --compile_depth GENERATE_INITIAL_GRAPH --pytest_directory_path forge/test/model_demos/high_prio/nlp/pytorch
```
