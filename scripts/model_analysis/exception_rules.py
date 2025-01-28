# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from utils import CompilerComponent, MatchingExceptionRule, MatchingCompilerComponentException
from exception_utils import collect_mlir_unsupported_ops, collect_error_msg_from_line

common_failure_matching_rules_list = [
    MatchingCompilerComponentException(
        CompilerComponent.FORGE,
        [
            MatchingExceptionRule(
                "forge_module evaluation", ["AssertionError", "Setting a tensor value of incorrect shape"]
            ),
            MatchingExceptionRule(
                "embedding indicies tensor",
                ["IndexError", "forge/forge/op/eval/forge/embedding.py", "index out of range in self"],
            ),
            MatchingExceptionRule(
                "post_initial_graph_passes",
                [
                    "RuntimeError",
                    "has_newstyle_interface(std::get<std::string>(type), false)",
                    "decomposing a type with old OpType interface, expects new OpType interface",
                ],
            ),
            MatchingExceptionRule(
                "lower_to_mlir",
                ["RuntimeError", "Found Unsupported operations while lowering from TTForge to TTIR in forward graph"],
                collect_mlir_unsupported_ops,
            ),
            MatchingExceptionRule(
                "lower_to_mlir",
                ["RuntimeError", "Unsupported data format during lowering from TTForge to TTIR"],
            ),
            MatchingExceptionRule(
                "mlir generation failure", ["RuntimeError", "Generated MLIR module failed verification"]
            ),
            MatchingExceptionRule(
                "Convert tt-forge attribute to an MLIR attribute", ["RuntimeError", "Unhandled attribute type"]
            ),
            MatchingExceptionRule("Runtime Datatype Unsupported", ["RuntimeError", "Unhandled dtype Bool"]),
            # Compiled model Runtime
            MatchingExceptionRule(
                "Runtime Datatype mismatch",
                ["RuntimeError", "Tensor", "data type mismatch: expected", "got"],
                collect_error_msg_from_line,
            ),
            MatchingExceptionRule(
                "Runtime Shape mismatch", ["RuntimeError", "Tensor", "shape mismatch: expected", "got"]
            ),
            MatchingExceptionRule(
                "Runtime stride mismatch",
                ["RuntimeError", "Tensor", "stride mismatch: expected", "got"],
                collect_error_msg_from_line,
            ),
            MatchingExceptionRule(
                "Runtime Input count mismatch", ["RuntimeError", "Input count mismatch: expected", "got"]
            ),
            MatchingExceptionRule(
                "post_const_eval_tensors", ["RuntimeError", "unsupported memory format option Contiguous"]
            ),
            MatchingExceptionRule(
                "TT-Metal vs Forge Output Dtype mismatch",
                ["TypeError: Dtype mismatch: framework_model.dtype", ", compiled_model.dtype"],
                collect_error_msg_from_line,
            ),
        ],
    ),
    MatchingCompilerComponentException(
        CompilerComponent.MLIR,
        [
            MatchingExceptionRule(
                "TTIR to TTNN Conv2dOpConversionPattern",
                [
                    "tt_forge_signal_handler",
                    "tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp",
                    "Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &)",
                    "adaptor.getPaddingBottom() == adaptor.getPaddingTop()",
                    "TTNN only supports padding height/width attributes. Thus, padding_top",
                    "must equal padding_bottom for the op to execute as expected",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.reshape mlir pipeline",
                [
                    "RuntimeError",
                    "'ttnn.reshape' op Shape attribute size must match output tensor rank",
                    "Failed to run MLIR compiler pass pipeline",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.maxpool2d mlir pipeline",
                [
                    "RuntimeError",
                    "ttnn.max_pool2d currently only supports an input type of bfloat16",
                    "Failed to run MLIR compiler pass pipeline",
                ],
            ),
            MatchingExceptionRule("mlir pipeline", ["RuntimeError", "Failed to run MLIR compiler pass pipeline"]),
            MatchingExceptionRule(
                "MLIR runtime ttnn ", ["tt::exception", "tt-mlir/runtime/lib/ttnn/runtime.cpp", "Unsupported data type"]
            ),
            MatchingExceptionRule(
                "mlir::AffineMap collapsedLinearAffineMap",
                [
                    "tt-mlir/lib/Dialect/TT/IR/TTOpsTypes.cpp",
                    "mlir::AffineMap collapsedLinearAffineMap",
                    "Dim does not participate in AffineMap RHS",
                ],
                collect_error_msg_from_line,
            ),
        ],
    ),
    MatchingCompilerComponentException(
        CompilerComponent.TT_METAL,
        [
            MatchingExceptionRule(
                "TT-Metal vs Forge Output Data mismatch",
                [
                    "ValueError",
                    "Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model",
                    ", compiled_model",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.tilize validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp",
                    "input_tensor_a.get_dtype() == DataType::BFLOAT16",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.tilize_with_val_padding validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp",
                    "input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32",
                    "Can only tilize bfloat16 or uint32 tensors",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.embedding validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp",
                    "weights.get_dtype() == DataType::BFLOAT16",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.embedding validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp",
                    "a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16",
                    "Input must be UINT32 or BFLOAT16",
                ],
            ),
            MatchingExceptionRule(
                "ttnn elementwise binary", ["RuntimeError", "BinaryOpType cannot be mapped to BcastOpMath"]
            ),
            MatchingExceptionRule(
                "ttnn elementwise binary",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp",
                    "ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.concat validation",
                ["RuntimeError", "Tile padding along concatenated dim", "not supported for concat yet"],
            ),
            MatchingExceptionRule(
                "ttnn.reshape validation",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp",
                    "input_tensor_a.get_dtype() == DataType::BFLOAT16",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.matmul",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp",
                    "Mt % per_core_M == 0",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.matmul",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp",
                    "Nt % per_core_N == 0",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.reshape",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp",
                    "new_volume == old_volume",
                    "Invalid arguments to reshape",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.reshape",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp",
                    "tensor_shape.rank() <= 4",
                    "Only up to 4D tensors",
                ],
            ),
            MatchingExceptionRule(
                "ttnn permute",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/permute/device/permute_device_operation.cpp",
                    "attributes.dims.back() == tensor_args.input_tensor.get_logical_shape().rank() - 1",
                    "Last dimension of permute must be the last dimension of the input tensor as page-breaking is not supported at the moment",
                ],
            ),
            MatchingExceptionRule(
                "ttnn.pad",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/data_movement/pad/pad.cpp",
                    "Tensor rank is not 4",
                ],
            ),
            MatchingExceptionRule(
                "TTNN tensor types",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/tensor/types.cpp",
                    "normalized_index >= 0 and normalized_index < rank",
                    "Index is out of bounds for the rank",
                ],
            ),
            MatchingExceptionRule(
                "TTNN tensor types",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/tensor/types.cpp",
                    "shape[cur_idx] == 1",
                    "Can't convert shape rank",
                ],
            ),
            MatchingExceptionRule("ttmetal allocations", ["RuntimeError", "Statically allocated circular buffers"]),
            MatchingExceptionRule(
                "ttmetal allocations",
                [
                    "RuntimeError",
                    "tt-metal/tt_metal/impl/allocator/allocator.cpp",
                    "Out of Memory: Not enough space to allocate",
                ],
            ),
            MatchingExceptionRule(
                "ttnn core",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp",
                    "logical_shape.rank() >= 2 && logical_shape.rank() <= 4",
                    "Only 2D, 3D, and 4D tensors are supported",
                ],
            ),
            MatchingExceptionRule(
                "ttnn softmax",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp",
                    "input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B",
                    "Inputs must be of bfloat16 or bfloat8_b type",
                ],
            ),
            MatchingExceptionRule(
                "ttnn unsqueeze_to_4D",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/core/core.cpp",
                    "Tensor rank is greater than 4",
                ],
            ),
            MatchingExceptionRule(
                "ttnn matmul",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp",
                    "(input_tensor_a.get_legacy_shape()[-1] / in0_tile_shape[1]) % program_config.in0_block_w == 0",
                    "Kt must be divisible by in0_block_w",
                ],
            ),
            MatchingExceptionRule(
                "tt-metal ncrisc build",
                [
                    "RuntimeError",
                    "tt-metal/tt_metal/impl/program/program.cpp",
                    "Failed to generate binaries for reader_conv_activations_padded_with_halo_3x3_weights_v2",
                    "ncrisc build failed",
                ],
            ),
            MatchingExceptionRule(
                "ttnn shared operation",
                [
                    "RuntimeError",
                    "ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp",
                    "(*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal.get_alignment(HalMemType::L1) == 0",
                    "Shard page size must currently have L1 aligned page size",
                ],
            ),
            MatchingExceptionRule(
                "ttnn pool",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp",
                    "in_ntiles_c % MAX_TILES_PER_REDUCTION == 0",
                    "input channels should be multiple of 8 tiles. General case TODO.",
                ],
            ),
            MatchingExceptionRule(
                "ttnn pool",
                [
                    "RuntimeError",
                    "tt-metal/ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp",
                    "input_shape[3] == 16",
                ],
            ),
        ],
    ),
]
