## Running Performance Benchmark Tests

You can use `forge/test/benchmark/benchmark.py` to run performance benchmark tests:

   ```bash
   python forge/test/benchmark/benchmark.py [options]
   ```

   **Available Options:**

   | Option | Short | Type | Default | Description |
   |--------|-------|------|---------|-------------|
   | `--model` | `-m` | string | *required* | Model to benchmark (e.g. bert, mnist_linear). The test file name without .py extension |
   | `--config` | `-c` | string | None | Model configuration to benchmark (e.g. tiny, base, large) |
   | `--training` | `-t` | flag | False | Benchmark training mode |
   | `--batch_size` | `-bs` | integer | 1 | Batch size, number of samples to process at once |
   | `--loop_count` | `-lp` | integer | 1 | Number of times to run the benchmark |
   | `--input_size` | `-isz` | integer | None | Input size of the input sample (if model supports variable input size) |
   | `--hidden_size` | `-hs` | integer | None | Hidden layer size (if model supports variable hidden size) |
   | `--output` | `-o` | string | None | Output JSON file to write results to. Results will be appended if file exists |
   | `--task` | `-ts` | string | "na" | Task to benchmark (e.g. classification, segmentation) |
   | `--data_format` | `-df` | string | "float32" | Data format (e.g. float32, bfloat16) |

   **Example:**

   ```bash
   python forge/test/benchmark/benchmark.py -m mobilenetv2_basic -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-fe-benchmark-e2e-mobilenetv2_basic.json
   ```

Alternatively, you can run specific model tests using `pytest`:

   ```bash
   pytest [model_path]
   ```

   **Example:**

   ```bash
   pytest -svv forge/test/benchmark/benchmark/models/yolo_v8.py
   ```
