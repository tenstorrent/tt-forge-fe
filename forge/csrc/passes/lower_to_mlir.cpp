// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "lower_to_mlir.hpp"

// Standard headers
#include <cstdint>
#include <stdexcept>
#include <string>

// TTForge headers
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/utils.hpp"
#include "graph_lib/node_types.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "utils/logger.hpp"

// MLIR headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#pragma clang diagnostic pop

// TTMLIR headers
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace 
{
using namespace tt;
/**
 * @brief Implementation of TT-MLIR emission from the TTForge graph.
 */

class MLIRGenerator
{
    public:
        /// Construct a new MLIRGenerator object.
        MLIRGenerator(mlir::MLIRContext &context) : builder_(&context)
        {
            init_lowering_handler_map();
        }

        /// Public API: Convert the TTForge graph into an MLIR module operation for TTIR.
        mlir::ModuleOp emit_mlir(graphlib::Graph *graph)
        {
            graphModule_ = mlir::ModuleOp::create(get_module_location(graph), "tt-forge-graph");
            graphModule_->setAttr(mlir::tt::SystemDescAttr::name,
                      mlir::tt::SystemDescAttr::getDefault(builder_.getContext()));
            builder_.setInsertionPointToStart(&graphModule_.getBodyRegion().front());

            {
                auto traversal_context = graphlib::get_subgraph_traversal_context<graphlib::SubgraphType::Forward>(graph);
                emit_mlir_function(graph);
            }

            if (graph->training())
            {
                auto traversal_context = graphlib::get_subgraph_traversal_context<graphlib::SubgraphType::Backward>(graph);
                emit_mlir_function(graph, "backward");
            }

            log_info(LogMLIRCompiler, "MLIR module generated successfully.");
            graphModule_.dump();

            /// Verify the module after we have finished constructing it, this will check
            /// the structural properties of the IR and invoke any specific verifiers we
            /// have on the TTIR operations.
            if (failed(mlir::verify(graphModule_))) 
            {
                graphModule_.emitError("module verification failed.");
                throw std::runtime_error("Generated MLIR module failed verification.");
            }

#ifdef DEBUG
            // Create a string to store the output
            std::string moduleStr;
            llvm::raw_string_ostream rso(moduleStr);

            // Print the MLIR module
            mlir::OpPrintingFlags printFlags;
            printFlags.enableDebugInfo();
            graphModule_.print(rso, printFlags);

            rso.flush();

            log_trace(LogMLIRCompiler, "MLIR module after lowering TT-Forge graph:\n{}", moduleStr);
#endif

            return graphModule_;
        }

    private:
        /// A "module" matches a TTForge graph: containing a single function to exectue.
        mlir::ModuleOp graphModule_;

        /// The builder is a helper class to create IR. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder_;

        /// The symbol table maintains a mapping between the names of ttforge nodes and their corresponding values in the current scope.
        /// Initially, the function arguments (model activations) are added to the symbol table.
        /// After evaluating each ttforge op node, the declare function adds a new entry to the symbol table for future reference.
        std::map<std::string, std::pair<mlir::Value, graphlib::Node*>> symbolTable_;

        /// Handler type for lowering ttforge operations to MLIR.
        using HandlerType = mlir::Value (MLIRGenerator::*)(tt::graphlib::Graph *, tt::graphlib::OpNode *);

        /// Map of lowering handlers for ttforge operations to MLIR.
        std::map<std::string, HandlerType> lowering_handler_map;

        /// Declares a variable in the current (only) scope.
        /// The declaration corresponds to exactly one operation node in the TTForge graph.
        void declare(graphlib::Node *node, mlir::Value value) {
            if (symbolTable_.find(node->name()) != symbolTable_.end())
            {
                throw std::runtime_error("Variable " + node->name() + " already declared in the current scope.");
            }
            
            symbolTable_[node->name()] = {value, node};
        }

        // Convert a TTForge attribute to an MLIR attribute.
        mlir::Attribute convert_to_mlir_attribute(const tt::BudaOpAttr& value) {
            return std::visit([this](auto&& arg) -> mlir::Attribute {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    return builder_.getStringAttr(arg);
                } else if constexpr (std::is_same_v<T, bool>) {
                    return builder_.getBoolAttr(arg);
                } else if constexpr (std::is_same_v<T, int>) {
                    return builder_.getSI32IntegerAttr(arg);
                } else if constexpr (std::is_same_v<T, float>) {
                    return builder_.getF32FloatAttr(arg);
                } else if constexpr (std::is_same_v<T, std::vector<int>>) {
                    llvm::SmallVector<mlir::Attribute> attributes;
                    for (auto& element : arg) {
                        attributes.push_back(builder_.getI32IntegerAttr(element));
                    }
                    return builder_.getArrayAttr(attributes);
                } else {
                    // If type not handled, throw an exception or handle it appropriately
                    throw std::runtime_error("Unhandled attribute type");
                }
            }, value);
        }

        /// Emit a new function in MLIR.
        /// A function represents a set of TTForge operations that are executed to produce output results.
        /// This function will generate the MLIR code for each TTForge operation in the graph and emit the return operation for the function.
        mlir::func::FuncOp emit_mlir_function(tt::graphlib::Graph *graph, std::string fn_name = "forward") {
            // Assemble the function arguments (inputs and parameters)
            llvm::SmallVector<mlir::Type> argument_types;
            llvm::SmallVector<graphlib::Node *> argument_nodes;
            
            symbolTable_.clear();

            // Add the graph inputs to the argument list
            for (auto *input: graph->ordered_module_inputs()) //for (auto *input : graph->nodes_by_type(tt::graphlib::kInput))
            {
                argument_nodes.push_back(input);
                argument_types.push_back(get_node_type(input));
            }

            // Add the graph parameters to the argument list
            for(auto *parameter: graph->get_parameter_nodes())
            {
                // Check whether the parameter is actually used in the current graph context,
                // for example when compiling model for training we will emit separate mlirs
                // for forward and backward subgraphs (via GraphTraversalContext).
                if (graph->data_users(parameter).empty())
                {
                    continue;
                }

                argument_nodes.push_back(parameter);
                argument_types.push_back(get_node_type(parameter));
            }

            // Assemble the function return values (outputs)
            llvm::SmallVector<mlir::Type> returns;
            auto output_nodes = graph->nodes([](const graphlib::Node *node) {
                return node->node_type() == tt::graphlib::NodeType::kOutput
                 || (node->node_type() == tt::graphlib::NodeType::kQueue && node->as<graphlib::QueueNode>()->is_grad_accumulator());
            });

            for (auto *output : output_nodes)
            {
                returns.push_back(get_node_type(output));
            }

            // Create the function and emit it in the MLIR module.
            auto funcType = builder_.getType<mlir::FunctionType>(mlir::TypeRange(argument_types), mlir::TypeRange(returns));
            auto func = builder_.create<mlir::func::FuncOp>(graphModule_.getLoc(), fn_name, funcType);
            
            // Set the function argument names
            for(size_t i = 0; i < argument_nodes.size(); i++)
            {
                graphlib::Node* argument_node = argument_nodes[i];
                llvm::SmallVector<mlir::NamedAttribute, 1> named_attributes;
                named_attributes.push_back(builder_.getNamedAttr("ttir.name", builder_.getStringAttr(argument_node->name())));
                func.setArgAttrs(i, named_attributes);
            }

            // Start the body of the function by creating an entry block.
            mlir::Block *entryBlock = func.addEntryBlock();

            // Declare function arguments in the symbol table
            for(auto namedValue: llvm::zip(argument_nodes, entryBlock->getArguments()))
            {
                graphlib::Node* argument_node = std::get<0>(namedValue);
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
                    continue;
                }

                log_trace(LogMLIRCompiler, "Emitting MLIR for node {}", node->name());

                tt::graphlib::OpNode *op_node = node->as<tt::graphlib::OpNode>();

                // Emit MLIR for the TTForge operation node
                mlir::Value opValue = emit_mlir_tt_forge_operation(graph, op_node);
                log_trace(LogMLIRCompiler, "Generated MLIR for node {} with value {}",
                    node->name(), covnert_mlir_value_to_string(opValue));
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

            // This is the first time we are visiting this TTForge node during the traversal of the graph using topological sort.
            // Therefore, we need to declare the result of this operation so that we can refer to it later if needed.
            declare(op_node, opResult);
            return opResult;
        }

        /// Emit an MLIR operation for a ttforge elementwise operation.
        template <typename TTIROp>
        mlir::Value emit_mlir_ttforge_op(tt::graphlib::Graph *graph, tt::graphlib::OpNode *op_node)
        {
            // Evaluate operation return type
            llvm::SmallVector<mlir::Type> return_types = get_mlir_type_range(op_node);

            // Evaluate operation operands: inputs and outputs per DPS
            llvm::SmallVector<mlir::Value> operands = get_mlir_operands(graph, op_node);

            // Evaluate opeartion attributes
            llvm::SmallVector<mlir::NamedAttribute> attributes;
            ::llvm::ArrayRef<::llvm::StringRef> operation_attributes = TTIROp::getAttributeNames();
            for(auto attribute_name: operation_attributes)
            {
                if (attribute_name == "operand_constraints")
                {
                    // Create operation constraint attributes
                    mlir::NamedAttribute operand_constraints_attribute = builder_.getNamedAttr(
                        "operand_constraints",
                    builder_.getArrayAttr(get_mlir_operand_constraint_attributes(graph, op_node)));
                    attributes.push_back(operand_constraints_attribute);
                }
                else if (attribute_name == mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr())
                {
                    // Create operation segment sizes attributes
                    mlir::NamedAttribute operand_segment_sizes_attribute = builder_.getNamedAttr(
                        mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(), 
                        builder_.getDenseI32ArrayAttr({
                            static_cast<int32_t>(graph->operands(op_node).size()), 
                        static_cast<int32_t>(1)
                        }));
                    attributes.push_back(operand_segment_sizes_attribute);
                }
            }

            for(const auto & attribute: op_node->op_type().named_attrs)
            {
                // convert atribute to mlir atribute
                auto mlir_atribute = convert_to_mlir_attribute(attribute.second);
                mlir::NamedAttribute named_attribute = builder_.getNamedAttr(
                                        attribute.first, mlir_atribute);
                attributes.push_back(named_attribute);
            }

            auto op = builder_.create<TTIROp>(
                get_tt_forge_operation_location(graph, op_node), 
                mlir::TypeRange(return_types),
                mlir::ValueRange(operands),
                attributes);

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
        llvm::SmallVector<mlir::Value> get_mlir_operands(
            tt::graphlib::Graph *graph,
            tt::graphlib::OpNode *op_node)
        {
            llvm::SmallVector<mlir::Value> operands;

            for (auto operand : graph->data_operands(op_node))
            {
                TT_ASSERT(symbolTable_.find(operand->name()) != symbolTable_.end(), "Operand " + operand->name() + "not found in symbol table.");
                operands.push_back(symbolTable_.at(operand->name()).first);
            }

            operands.push_back(emit_mlir_empty_tensor(graph, op_node));
            return operands;
        }

        // Get the MLIR operand constraint attributes for a TTForge operation.
        llvm::SmallVector<mlir::Attribute> get_mlir_operand_constraint_attributes(
            tt::graphlib::Graph *graph,
            tt::graphlib::OpNode *op_node)
        {
            llvm::SmallVector<mlir::Attribute> operand_constraints;

            for ([[maybe_unused]] auto& operand: graph->operands(op_node))
            {
                mlir::Attribute operand_constraint_attribute = builder_.getAttr<mlir::tt::OperandConstraintAttr>(
                           mlir::tt::OperandConstraint::AnyDevice);
                operand_constraints.push_back(operand_constraint_attribute);
            }

            for ([[maybe_unused]] auto& user: graph->data_users(op_node))
            {
                mlir::Attribute operand_constraint_attribute = builder_.getAttr<mlir::tt::OperandConstraintAttr>(
                           mlir::tt::OperandConstraint::AnyDevice);
                operand_constraints.push_back(operand_constraint_attribute);           
            }

            return operand_constraints;
        }

        /// Emit an MLIR operation for an empty tensor.
        mlir::Value emit_mlir_empty_tensor(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
        {
            llvm::SmallVector<int64_t> shape_vec;

            for(auto dim : node->shape().as_vector())
            {
                shape_vec.push_back((int64_t)dim);
            }

            return builder_.create<mlir::tensor::EmptyOp>(
                get_tt_forge_operation_location(graph, node),
                shape_vec,
                get_data_type(node));
        }

        /// Emit the return operation for the function.
        void emit_mlir_return_op(tt::graphlib::Graph *graph)
        {
            // Assemble the function return values (outputs)
            llvm::SmallVector<mlir::Value> returnValues;

            auto output_nodes = graph->nodes([](const graphlib::Node *node) {
                return node->node_type() == tt::graphlib::NodeType::kOutput
                 || (node->node_type() == tt::graphlib::NodeType::kQueue && node->as<graphlib::QueueNode>()->is_grad_accumulator());
            });

            for (auto *output : output_nodes)
            {
                TT_ASSERT(graph->data_operands(output).size() == 1, "Output node must have exactly one operand.");
                auto output_operand = graph->data_operands(output)[0];
                auto outputValue = symbolTable_[output_operand->name()].first;
                returnValues.push_back(outputValue);
            }

            builder_.create<mlir::func::ReturnOp>(
                builder_.getUnknownLoc(), 
                mlir::ValueRange(returnValues));
        }

        /// Get the MLIR data type for a TTForge node.
        mlir::Type get_data_type(graphlib::Node *node)
        {
            switch (node->output_df())
            {
                case tt::DataFormat::Float32:
                    return builder_.getF32Type();
                case tt::DataFormat::Float16_b:
                    return builder_.getBF16Type();
                case tt::DataFormat::Float16:
                    return builder_.getF16Type();
                case tt::DataFormat::Int8:
                    return builder_.getI8Type();
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

        /// Get the location for a module.
        mlir::Location get_module_location(tt::graphlib::Graph *graph)
        {
            return mlir::FileLineColLoc::get(
                builder_.getContext(), graph->name(), graph->id(), 0);
        }

        /// Get the simple location for a node in a format "graph_name", (graph_id), (node_id)
        mlir::Location get_node_location(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
        {
            return mlir::FileLineColLoc::get(
                builder_.getContext(), graph->name(), graph->id(), node->id());
        }

        /// Get the location for a TTForge operation. The location is a combination of the operation name and the node location.
        mlir::Location get_tt_forge_operation_location(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
        {
            return mlir::NameLoc::get(
                builder_.getStringAttr(node->name()), get_node_location(graph, node));
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
        
        /// Initialize lowering handler map
        void init_lowering_handler_map()
        {
            lowering_handler_map["add"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::AddOp>;
            lowering_handler_map["embedding"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::EmbeddingOp>;
            lowering_handler_map["matmul"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MatmulOp>;
            lowering_handler_map["multiply"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MultiplyOp>;
            lowering_handler_map["reduce_avg"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::MeanOp>;
            lowering_handler_map["reduce_sum"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SumOp>;
            lowering_handler_map["relu"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::ReluOp>;
            lowering_handler_map["softmax"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SoftmaxOp>;
            lowering_handler_map["sqrt"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SqrtOp>;
            lowering_handler_map["subtract"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::SubtractOp>;
            lowering_handler_map["transpose"] = &MLIRGenerator::emit_mlir_ttforge_op<mlir::tt::ttir::TransposeOp>;
        }
};
}
namespace tt::passes
{
    /// Public API for generating MLIR from the TTForge graph.
     mlir::OwningOpRef<mlir::ModuleOp> lower_to_mlir(graphlib::Graph * graph, mlir::MLIRContext& context)
    {
        return MLIRGenerator(context).emit_mlir(graph);
    }
}