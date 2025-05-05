// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "lower_to_mlir.hpp"

// Standard headers
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utils/assert.hpp>
#include <variant>

// TTForge headers
#include "forge_graph_module.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "passes/extract_unique_op_configuration.hpp"
#include "utils/logger.hpp"

// MLIR headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#pragma clang diagnostic pop

// TTMLIR headers
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

// Reportify headers
#include "reportify/reportify.hpp"

namespace
{
using namespace tt;
/**
 * @brief Implementation of TT-MLIR emission from the Forge module (set of graphs).
 */

enum class TargetType
{
    SourceType,
    UI32Attr,
    I64Attr,
    I32Attr,
    F32Attr,
    I32ArrayAttr,
    DenseI64ArrayAttr,
    DenseI32ArrayAttr,
};

struct AttributeRemap
{
    std::optional<std::string> new_name;  // New name for the attribute
    TargetType target_type_value;         // Target type conversion

    AttributeRemap(const std::optional<std::string> &name = std::nullopt, TargetType type = TargetType::SourceType) :
        new_name(name), target_type_value(type)
    {
    }
};

class AttributeMapper
{
   public:
    AttributeMapper() { initialize_default_mappings(); }

    // Add mapping for a specific op's attribute
    void add_op_mapping(const std::string &op_name, const std::string &attr_name, const AttributeRemap &remap)
    {
        mappings_[op_name][attr_name] = remap;
    }

    // Get the mapped name and target type for an attribute
    std::pair<std::string, TargetType> get_mapped_name_and_type(
        const std::string &op_name, const std::string &attr_name) const
    {
        auto op_it = mappings_.find(op_name);
        if (op_it != mappings_.end())
        {
            auto attr_it = op_it->second.find(attr_name);
            if (attr_it != op_it->second.end())
            {
                const auto &remap = attr_it->second;
                return {remap.new_name.value_or(attr_name), remap.target_type_value};
            }
        }
        return {attr_name, TargetType::SourceType};
    }

   private:
    // Mapping storage: op_name -> (attr_name -> remap)
    std::map<std::string, std::map<std::string, AttributeRemap>> mappings_;

    void initialize_default_mappings()
    {
        // Sort the mappings in lexicographical order

        // argmax
        add_op_mapping("argmax", "dim", AttributeRemap("dim_arg", TargetType::I32ArrayAttr));

        // conv2d_transpose
        add_op_mapping("conv2d_transpose", "dilation", AttributeRemap(std::nullopt, TargetType::DenseI32ArrayAttr));
        add_op_mapping("conv2d_transpose", "groups", AttributeRemap(std::nullopt, TargetType::I32Attr));
        add_op_mapping(
            "conv2d_transpose", "output_padding", AttributeRemap(std::nullopt, TargetType::DenseI32ArrayAttr));
        add_op_mapping("conv2d_transpose", "padding", AttributeRemap(std::nullopt, TargetType::DenseI32ArrayAttr));
        add_op_mapping("conv2d_transpose", "stride", AttributeRemap(std::nullopt, TargetType::DenseI32ArrayAttr));

        // conv2d
        add_op_mapping("conv2d", "dilation", AttributeRemap(std::nullopt, TargetType::DenseI32ArrayAttr));
        add_op_mapping("conv2d", "groups", AttributeRemap(std::nullopt, TargetType::I32Attr));
        add_op_mapping("conv2d", "padding", AttributeRemap(std::nullopt, TargetType::DenseI32ArrayAttr));
        add_op_mapping("conv2d", "stride", AttributeRemap(std::nullopt, TargetType::DenseI32ArrayAttr));

        // cumsum
        add_op_mapping("cumsum", "dim", AttributeRemap(std::nullopt, TargetType::I64Attr));

        // index
        add_op_mapping("index", "dim", AttributeRemap(std::nullopt, TargetType::I32Attr));
        add_op_mapping("index", "start", AttributeRemap("begin", TargetType::I32Attr));
        add_op_mapping("index", "stop", AttributeRemap("end", TargetType::I32Attr));
        add_op_mapping("index", "stride", AttributeRemap("step", TargetType::I32Attr));

        // reduce_avg
        add_op_mapping("reduce_avg", "dim", AttributeRemap("dim_arg"));

        // repeat_interleave
        add_op_mapping("repeat_interleave", "repeats", AttributeRemap(std::nullopt, TargetType::UI32Attr));

        // repeat
        add_op_mapping("repeat", "repeats", AttributeRemap("repeat_dimensions", TargetType::DenseI64ArrayAttr));
        add_op_mapping("pad", "padding", AttributeRemap("padding", TargetType::DenseI32ArrayAttr));
        add_op_mapping("pad", "value", AttributeRemap("value", TargetType::F32Attr));

        // Add more default mappings here
    }
};

class MLIRGenerator
{
   public:
    /// Construct a new MLIRGenerator object.
    MLIRGenerator(mlir::MLIRContext &context) : builder_(&context) { init_lowering_handler_map(); }

    /// Public API: Convert the ForgeGraphModule into an MLIR module operation for TTIR.
    mlir::ModuleOp emit_mlir(tt::ForgeGraphModule &module)
    {
        graphModule_ = mlir::ModuleOp::create(get_module_location(module), module.name());
        graphModule_->setAttr(
            mlir::tt::SystemDescAttr::name, mlir::tt::SystemDescAttr::getDefault(builder_.getContext()));
        builder_.setInsertionPointToStart(&graphModule_.getBodyRegion().front());

        // Collect all the supported TTIR operations
        std::vector<std::string> supported_ops;
        std::transform(
            lowering_handler_map.begin(),
            lowering_handler_map.end(),
            std::back_inserter(supported_ops),
            [](const std::pair<std::string, HandlerType> &ttir_op_pair) { return ttir_op_pair.first; });

        // Emit MLIR functions for each graph in the module.
        for (auto graph : module.graphs())
        {
            // Verifies if any unsupported operations exist in the graph.
            // If found, an error is thrown, listing all unsupported operations
            // along with their unique configurations within the graph.
            auto unsupported_op_shapes_attrs = tt::passes::extract_unique_op_configuration(graph, supported_ops);

            if (!unsupported_op_shapes_attrs.empty())
            {
                log_error(
                    "Found Unsupported operations while lowering from TTForge to TTIR in {} graph", graph->name());
                tt::passes::print_unique_op_configuration(
                    unsupported_op_shapes_attrs, "Unsupported Ops at: RUN_MLIR_COMPILER stage");
                throw std::runtime_error(
                    "Found Unsupported operations while lowering from TTForge to TTIR in " + graph->name() + " graph");
            }

            emit_mlir_function(graph, graph->name());
        }

        /// Verify the module after we have finished constructing it, this will check
        /// the structural properties of the IR and invoke any specific verifiers we
        /// have on the TTIR operations.
        if (failed(mlir::verify(graphModule_)))
        {
            graphModule_.emitError("module verification failed.");
            throw std::runtime_error("Generated MLIR module failed verification.");
        }

        log_info(LogMLIRCompiler, "MLIR module generated successfully.");
        graphModule_.dump();

        // save what's dumped to a file named "{file_name}.mlir"
        reportify::dump_mlir("ttir", graphModule_.getNameAttr().getValue().str(), graphModule_.getOperation());

#ifdef DEBUG
        // Create a string to store the output
        std::string moduleStr;
        llvm::raw_string_ostream rso(moduleStr);

        // Print the MLIR module
        mlir::OpPrintingFlags printFlags;
        printFlags.enableDebugInfo();
        graphModule_.print(rso, printFlags);

        rso.flush();

        log_trace(LogMLIRCompiler, "MLIR module after lowering ForgeGraphModule:\n{}", moduleStr);
#endif

        return graphModule_;
    }

   private:
    /// A "module" matches the set of graphs contained in ForgeGraphModule.
    /// Where each graph will lower into a separate MLIR function inside the module.
    mlir::ModuleOp graphModule_;

    /// The builder is a helper class to create IR. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder_;

    /// The symbol table maintains a mapping between the names of ttforge nodes and their corresponding values in the
    /// current scope. Initially, the function arguments (model activations) are added to the symbol table. After
    /// evaluating each ttforge op node, the declare function adds a new entry to the symbol table for future reference.
    std::map<std::string, std::pair<mlir::Value, graphlib::Node *>> symbolTable_;

    /// Handler type for lowering ttforge operations to MLIR.
    using HandlerType = mlir::Value (MLIRGenerator::*)(tt::graphlib::Graph *, tt::graphlib::OpNode *);

    /// Map of lowering handlers for ttforge operations to MLIR.
    std::map<std::string, HandlerType> lowering_handler_map;

    /// Attribute mapper for handling attribute conversions
    static AttributeMapper attr_mapper_;

    /// Declares a variable in the current (only) scope.
    /// The declaration corresponds to exactly one operation node in the TTForge graph.
    void declare(graphlib::Node *node, mlir::Value value)
    {
        if (symbolTable_.find(node->name()) != symbolTable_.end())
        {
            throw std::runtime_error("Variable " + node->name() + " already declared in the current scope.");
        }

        log_trace(LogMLIRCompiler, "Declaring {} in the current scope.", node->name());

        symbolTable_[node->name()] = {value, node};
    }

    // Convert a TTForge attribute to an MLIR attribute.
    mlir::Attribute convert_to_mlir_attribute(const tt::ForgeOpAttr &value, TargetType target_type)
    {
        if (target_type != TargetType::SourceType)
        {
            // Convert the attribute to the target type
            switch (target_type)
            {
                case TargetType::UI32Attr:
                    TT_ASSERT(std::get<int>(value) >= 0, "Value must be an >= 0 for conversion to uint32");
                    return builder_.getUI32IntegerAttr(static_cast<uint32_t>(std::get<int>(value)));
                case TargetType::I32Attr: return builder_.getI32IntegerAttr(static_cast<int32_t>(std::get<int>(value)));
                case TargetType::I64Attr: return builder_.getI64IntegerAttr(static_cast<int64_t>(std::get<int>(value)));
                case TargetType::F32Attr: return builder_.getF32FloatAttr(static_cast<float>(std::get<float>(value)));
                case TargetType::DenseI64ArrayAttr:
                    return builder_.getDenseI64ArrayAttr(std::vector<int64_t>(
                        std::get<std::vector<int>>(value).begin(), std::get<std::vector<int>>(value).end()));
                case TargetType::DenseI32ArrayAttr:
                    return builder_.getDenseI32ArrayAttr(std::vector<int32_t>(
                        std::get<std::vector<int>>(value).begin(), std::get<std::vector<int>>(value).end()));
                case TargetType::I32ArrayAttr:
                    return std::visit(
                        [this](auto &&arg) -> mlir::Attribute
                        {
                            using T = std::decay_t<decltype(arg)>;
                            // if we have a single int, convert it to an array of size 1
                            if constexpr (std::is_same_v<T, int>)
                            {
                                return builder_.getI32ArrayAttr({arg});
                            }
                            // if we have a vector of ints, convert it to an array
                            else if constexpr (std::is_same_v<T, std::vector<int>>)
                            {
                                return builder_.getI32ArrayAttr(llvm::SmallVector<int>(arg.begin(), arg.end()));
                            }
                            else
                            {
                                // If type not handled, throw an exception
                                throw std::runtime_error("Unhandled attribute type");
                            }
                        },
                        value);
                default:
                    // If type not handled, throw an exception
                    throw std::runtime_error("Unhandled target type conversion");
            }
        }
        return std::visit(
            [this](auto &&arg) -> mlir::Attribute
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>)
                {
                    return builder_.getStringAttr(arg);
                }
                else if constexpr (std::is_same_v<T, bool>)
                {
                    return builder_.getBoolAttr(arg);
                }
                else if constexpr (std::is_same_v<T, int>)
                {
                    return builder_.getSI32IntegerAttr(arg);
                }
                else if constexpr (std::is_same_v<T, float>)
                {
                    return builder_.getF32FloatAttr(arg);
                }
                else if constexpr (std::is_same_v<T, std::vector<int>>)
                {
                    llvm::SmallVector<mlir::Attribute> attributes;
                    for (auto &element : arg)
                    {
                        attributes.push_back(builder_.getI32IntegerAttr(element));
                    }
                    return builder_.getArrayAttr(attributes);
                }
                else
                {
                    // If type not handled, throw an exception or handle it appropriately
                    throw std::runtime_error("Unhandled attribute type");
                }
            },
            value);
    }

    /// Emit a new function in MLIR.
    /// A function represents a set of TTForge operations that are executed to produce output results.
    /// This function will generate the MLIR code for each TTForge operation in the graph and emit the return operation
    /// for the function.
    mlir::func::FuncOp emit_mlir_function(tt::graphlib::Graph *graph, std::string fn_name = "forward")
    {
        log_info("Emitting mlir for function {}", fn_name);
        // Assemble the function arguments (inputs and parameters)
        llvm::SmallVector<mlir::Type> argument_types;
        llvm::SmallVector<graphlib::Node *> argument_nodes;

        symbolTable_.clear();

        // Add the graph inputs to the argument list.
        for (auto *input : graph->ordered_module_inputs())
        {
            log_trace(LogMLIRCompiler, "Adding input {} to the argument list.", input->name());

            argument_nodes.push_back(input);
            argument_types.push_back(get_node_type(input));
        }

        // Add the graph constants to the argument list.
        for (auto *constant : graph->get_constant_nodes())
        {
            log_trace(LogMLIRCompiler, "Adding constant {} to the argument list.", constant->name());

            argument_nodes.push_back(constant);
            argument_types.push_back(get_node_type(constant));
        }

        // Add the graph parameters to the argument list.
        // Both optimizer parameters and regular parameters are added.
        auto opt_params = graph->get_optimizer_parameter_nodes();
        auto params = graph->get_parameter_nodes();
        params.insert(params.end(), opt_params.begin(), opt_params.end());
        for (auto *parameter : params)
        {
            log_trace(LogMLIRCompiler, "Adding parameter {} to the argument list.", parameter->name());

            argument_nodes.push_back(parameter);
            argument_types.push_back(get_node_type(parameter));
        }

        // Assemble the function return values (outputs)
        llvm::SmallVector<mlir::Type> returns;
        auto output_nodes = graph->ordered_module_outputs();
        for (auto *output : output_nodes)
        {
            log_trace(LogMLIRCompiler, "Adding output {} to the return list.", output->name());
            returns.push_back(get_node_type(output));
        }

        // Create the function and emit it in the MLIR module.
        auto funcType = builder_.getType<mlir::FunctionType>(mlir::TypeRange(argument_types), mlir::TypeRange(returns));
        auto func = builder_.create<mlir::func::FuncOp>(graphModule_.getLoc(), fn_name, funcType);

        // Set the function argument names.
        for (size_t i = 0; i < argument_nodes.size(); i++)
        {
            graphlib::Node *argument_node = argument_nodes[i];
            llvm::SmallVector<mlir::NamedAttribute, 1> named_attributes;
            named_attributes.push_back(
                builder_.getNamedAttr("ttir.name", builder_.getStringAttr(argument_node->name())));
            named_attributes.push_back(builder_.getNamedAttr("tt.argument_type", get_argument_type(argument_node)));
            func.setArgAttrs(i, named_attributes);
            log_trace(LogMLIRCompiler, "Set argument name {} for function argument {}.", argument_node->name(), i);
        }

        // Set the return value names.
        for (size_t i = 0; i < output_nodes.size(); i++)
        {
            graphlib::Node *output_node = output_nodes[i];
            llvm::SmallVector<mlir::NamedAttribute, 1> named_attributes;
            named_attributes.push_back(builder_.getNamedAttr("ttir.name", builder_.getStringAttr(output_node->name())));
            func.setResultAttrs(i, named_attributes);
            log_trace(LogMLIRCompiler, "Set name {} for return value {}.", output_node->name(), i);
        }

        // Start the body of the function by creating an entry block.
        mlir::Block *entryBlock = func.addEntryBlock();

        // Declare function arguments in the symbol table
        for (auto namedValue : llvm::zip(argument_nodes, entryBlock->getArguments()))
        {
            graphlib::Node *argument_node = std::get<0>(namedValue);
            mlir::BlockArgument arg = std::get<1>(namedValue);
            declare(argument_node, arg);
        }

        // Set the insertion point in the builder to the beginning of the function
        // body, it will be used throughout the codegen to create operations in this
        // function.
        auto savedInsertionPoint = builder_.saveInsertionPoint();
        builder_.setInsertionPointToStart(entryBlock);

        // Walk the graph in topological order and generate MLIR for each TTForge operation
        // node in the graph. For each new operation result, declare it in the symbol table.
        for (auto *node : graphlib::topological_sort(*graph))
        {
            // Skip if the node isn't TTForge operation
            if (node->node_type() != tt::graphlib::NodeType::kPyOp)
            {
                log_trace(LogMLIRCompiler, "Skipping node {} as it is not a TTForge operation.", node->name());
                continue;
            }
            log_trace(LogMLIRCompiler, "Emitting MLIR for node {}", node->name());

            tt::graphlib::OpNode *op_node = node->as<tt::graphlib::OpNode>();

            // Emit MLIR for the TTForge operation node
            mlir::Value opValue = emit_mlir_tt_forge_operation(graph, op_node);
            log_trace(
                LogMLIRCompiler,
                "Generated MLIR for node {} with value {}",
                node->name(),
                covnert_mlir_value_to_string(opValue));
        }
        emit_mlir_return_op(graph);

        // Restore the saved insertion point.
        builder_.restoreInsertionPoint(savedInsertionPoint);

        return func;
    }

    /// Emit an MLIR operation for a TTForge node.
    mlir::Value emit_mlir_tt_forge_operation(tt::graphlib::Graph *graph, tt::graphlib::OpNode *op_node)
    {
        auto handler = lowering_handler_map.find(op_node->op_name());
        // There is no known lowering handler for this operation. Report error.
        if (handler == lowering_handler_map.end())
        {
            log_error("Unsupported operation for lowering from TTForge to TTIR: {}", op_node->op_name());
            throw std::runtime_error("Unsupported operation for lowering from TTForge to TTIR: " + op_node->op_name());
        }

        // Call the handler to lower the TTForge op to MLIR
        mlir::Value opResult = (this->*(handler->second))(graph, op_node);

        // This is the first time we are visiting this TTForge node during the traversal of the graph using topological
        // sort. Therefore, we need to declare the result of this operation so that we can refer to it later if needed.
        declare(op_node, opResult);
        return opResult;
    }

    /// Emit an MLIR operation for a ttforge operation.
    template <typename TTIROp>
    mlir::Value emit_mlir_ttforge_op(tt::graphlib::Graph *graph, tt::graphlib::OpNode *op_node)
    {
        // Evaluate operation return type
        llvm::SmallVector<mlir::Type> return_types = get_mlir_type_range(op_node);

        // Evaluate operation operands: inputs and outputs per DPS
        llvm::SmallVector<mlir::Value> operands = get_mlir_operands(graph, op_node);

        // Map forge to MLIR attributes for this operation.
        llvm::SmallVector<mlir::NamedAttribute> mlir_attributes;
        for (const auto &[name, value] : op_node->op_type().named_attrs)
        {
            auto [mapped_name, target_type] = attr_mapper_.get_mapped_name_and_type(op_node->op_name(), name);

            mlir_attributes.push_back(
                builder_.getNamedAttr(mapped_name, convert_to_mlir_attribute(value, target_type)));
        }

        auto op = builder_.create<TTIROp>(
            get_tt_forge_operation_location(graph, op_node),
            mlir::TypeRange(return_types),
            mlir::ValueRange(operands),
            mlir_attributes);

        return op.getOperation()->getResult(0);
    }

    // Get the TT-MLIR type for a TTForge operation.
    llvm::SmallVector<mlir::Type> get_mlir_type_range(tt::graphlib::OpNode *op_node)
    {
        llvm::SmallVector<mlir::Type> return_type_vector;
        return_type_vector.push_back(get_node_type(op_node));
        return return_type_vector;
    }

    // All operands must be present in the symbol table, since we are
    // traversing the TTForge graph using topological sort. We iterate over the
    // operands of the current node and retrieve their corresponding values
    // from the symbol table.
    llvm::SmallVector<mlir::Value> get_mlir_operands(tt::graphlib::Graph *graph, tt::graphlib::OpNode *op_node)
    {
        llvm::SmallVector<mlir::Value> operands;

#ifdef DEBUG
        // Log all values from symbolTable_
        log_trace(LogMLIRCompiler, "Logging all keys from symbolTable_");
        for (const auto &entry : symbolTable_)
        {
            log_trace(LogMLIRCompiler, "Key: {}", entry.first);
        }
#endif

        for (auto operand : graph->data_operands(op_node))
        {
            TT_ASSERT(
                symbolTable_.find(operand->name()) != symbolTable_.end(),
                "Operand " + operand->name() + " not found in symbol table.");
            operands.push_back(symbolTable_.at(operand->name()).first);
        }

        operands.push_back(emit_mlir_empty_tensor(graph, op_node));
        return operands;
    }

    /// Emit an MLIR operation for an empty tensor.
    mlir::Value emit_mlir_empty_tensor(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
    {
        llvm::SmallVector<int64_t> shape_vec;

        for (auto dim : node->shape().as_vector())
        {
            shape_vec.push_back((int64_t)dim);
        }

        return builder_.create<mlir::tt::ttir::EmptyOp>(
            get_tt_forge_operation_location(graph, node), shape_vec, get_data_type(node));
    }

    /// Emit the return operation for the function.
    void emit_mlir_return_op(tt::graphlib::Graph *graph)
    {
        // Assemble the function return values (outputs)
        llvm::SmallVector<mlir::Value> returnValues;

        auto output_nodes = graph->ordered_module_outputs();
        for (auto *output : output_nodes)
        {
            TT_ASSERT(
                graph->data_operands(output).size() == 1,
                "Output node " + output->name() + " must have exactly one operand.");
            auto output_operand = graph->data_operands(output)[0];
            auto outputValue = symbolTable_[output_operand->name()].first;
            returnValues.push_back(outputValue);
        }

        builder_.create<mlir::func::ReturnOp>(builder_.getUnknownLoc(), mlir::ValueRange(returnValues));
    }

    /// Get the MLIR data type for a TTForge node.
    mlir::Type get_data_type(graphlib::Node *node)
    {
        switch (node->output_df())
        {
            case tt::DataFormat::Float32: return builder_.getF32Type();
            case tt::DataFormat::Float16_b: return builder_.getBF16Type();
            case tt::DataFormat::Float16: return builder_.getF16Type();
            case tt::DataFormat::Int32: return builder_.getI32Type();
            case tt::DataFormat::Int8: return builder_.getI8Type();
            case tt::DataFormat::RawUInt8: return builder_.getIntegerType(8, false);
            default:
                log_error("Unsupported data format during lowering from TTForge to TTIR: {}", node->output_df());
                TT_ASSERT(false);
        }

        // TODO add all supported types in switch
        return builder_.getF32Type();
    }

    /// Get the MLIR type for a TTForge node.
    mlir::Type get_node_type(graphlib::Node *node)
    {
        std::vector<int64_t> shape_vec;

        for (auto dim : node->shape().as_vector())
        {
            shape_vec.push_back((int64_t)dim);
        }

        return mlir::RankedTensorType::get(shape_vec, get_data_type(node));
    }

    mlir::tt::ArgumentTypeAttr get_argument_type(graphlib::Node *node)
    {
        auto input_node = node->as<graphlib::InputNode>();
        switch (input_node->input_type())
        {
            case tt::graphlib::InputNodeType::Activation:
            case tt::graphlib::InputNodeType::Loss:
            case tt::graphlib::InputNodeType::Target:
            case tt::graphlib::InputNodeType::Gradient:
            case tt::graphlib::InputNodeType::Accumulator:
                return mlir::tt::ArgumentTypeAttr::get(builder_.getContext(), mlir::tt::ArgumentType::Input);
            case tt::graphlib::InputNodeType::Parameter:
            case tt::graphlib::InputNodeType::OptimizerParameter:
                return mlir::tt::ArgumentTypeAttr::get(builder_.getContext(), mlir::tt::ArgumentType::Parameter);
            case tt::graphlib::InputNodeType::Constant:
                return mlir::tt::ArgumentTypeAttr::get(builder_.getContext(), mlir::tt::ArgumentType::Constant);
            default: throw std::runtime_error("Unknown input node type - cannot map to MLIR argument type");
        }
    }

    /// Get the location for a module.
    mlir::Location get_module_location(tt::ForgeGraphModule &module)
    {
        return mlir::FileLineColLoc::get(builder_.getContext(), module.name(), 0, 0);
    }

    /// Get the node location in format "source_location", (graph_id), (node_id)
    mlir::Location get_node_location(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
    {
        TT_ASSERT(graph != nullptr);
        TT_ASSERT(node != nullptr);

        const graphlib::TaggedNode *tagged_node = node->as<graphlib::TaggedNode>();

        // Get source location from layer tag if available, otherwise use node name
        std::string source_location = node->name();
        if (tagged_node && tagged_node->has_tag("layer"))
        {
            source_location = std::get<std::string>(tagged_node->tag_value("layer")) + "/" + node->name();
        }

        // Create and return FileLineColLoc with the collected information
        return mlir::FileLineColLoc::get(builder_.getContext(), source_location, graph->id(), node->id());
    }

    /// Get the location for a TTForge operation. The location is a combination of the operation name and the node
    /// location.
    mlir::Location get_tt_forge_operation_location(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
    {
        return mlir::NameLoc::get(builder_.getStringAttr(graph->name()), get_node_location(graph, node));
    }

    /// Convert an MLIR value to a string.
    std::string covnert_mlir_value_to_string(mlir::Value &value)
    {
        std::string string_value;
        llvm::raw_string_ostream os(string_value);
        os << value;
        os.flush();
        return string_value;
    }

    /// Initialize lowering handler map, keep in lexicographical order
    void init_lowering_handler_map()
    {
        lowering_handler_map["abs"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::AbsOp>;
        lowering_handler_map["add"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::AddOp>;
        lowering_handler_map["argmax"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ArgMaxOp>;
        lowering_handler_map["cast"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::TypecastOp>;
        lowering_handler_map["clip"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ClampScalarOp>;
        lowering_handler_map["concatenate"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ConcatOp>;
        lowering_handler_map["conv2d"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::Conv2dOp>;
        lowering_handler_map["cosine"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::CosOp>;
        lowering_handler_map["cumsum"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::CumSumOp>;
        lowering_handler_map["embedding_bw"] =
            &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::EmbeddingBackwardOp>;
        lowering_handler_map["embedding"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::EmbeddingOp>;
        lowering_handler_map["equal"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::EqualOp>;
        lowering_handler_map["exp"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ExpOp>;
        lowering_handler_map["gelu"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::GeluOp>;
        lowering_handler_map["greater_equal"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::GreaterEqualOp>;
        lowering_handler_map["greater"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::GreaterThanOp>;
        lowering_handler_map["index"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::IndexOp>;
        lowering_handler_map["leaky_relu"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::LeakyReluOp>;
        lowering_handler_map["less"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::LessEqualOp>;
        lowering_handler_map["log"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::LogOp>;
        lowering_handler_map["matmul"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MatmulOp>;
        lowering_handler_map["max_pool2d"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MaxPool2dOp>;
        lowering_handler_map["maximum"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MaximumOp>;
        lowering_handler_map["multiply"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MultiplyOp>;
        lowering_handler_map["not_equal"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::NotEqualOp>;
        lowering_handler_map["reciprocal"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ReciprocalOp>;
        lowering_handler_map["reduce_avg"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MeanOp>;
        lowering_handler_map["reduce_max"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MaxOp>;
        lowering_handler_map["reduce_sum"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SumOp>;
        lowering_handler_map["relu"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ReluOp>;
        lowering_handler_map["remainder"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::RemainderOp>;
        lowering_handler_map["repeat_interleave"] =
            &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::RepeatInterleaveOp>;
        lowering_handler_map["repeat"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::RepeatOp>;
        lowering_handler_map["conv2d_transpose"] =
            &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ConvTranspose2dOp>;
        lowering_handler_map["reshape"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ReshapeOp>;
        lowering_handler_map["select"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SelectOp>;
        lowering_handler_map["sigmoid"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SigmoidOp>;
        lowering_handler_map["sine"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SinOp>;
        lowering_handler_map["softmax"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SoftmaxOp>;
        lowering_handler_map["sqrt"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SqrtOp>;
        lowering_handler_map["squeeze"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SqueezeOp>;
        lowering_handler_map["subtract"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SubtractOp>;
        lowering_handler_map["tanh"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::TanhOp>;
        lowering_handler_map["transpose"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::TransposeOp>;
        lowering_handler_map["unsqueeze"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::UnsqueezeOp>;
        lowering_handler_map["upsample2d"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::Upsample2dOp>;
        lowering_handler_map["pad"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::PadOp>;
    }
};

AttributeMapper MLIRGenerator::attr_mapper_;
}  // namespace
namespace tt::passes
{
/// Public API for generating MLIR from the Forge module (set of graphs).
mlir::OwningOpRef<mlir::ModuleOp> lower_to_mlir(tt::ForgeGraphModule &module, mlir::MLIRContext &context)
{
    return MLIRGenerator(context).emit_mlir(module);
}
}  // namespace tt::passes
