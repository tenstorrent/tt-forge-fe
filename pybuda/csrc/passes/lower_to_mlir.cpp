// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "lower_to_mlir.hpp"

// Standard headers
#include <stdexcept>
#include <string>

// PyBuda headers
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/utils.hpp"
#include "graph_lib/node_types.hpp"
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
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace 
{
using namespace tt;
/**
 * @brief Implementation of TT-MLIR emission from the PyBuda graph.
 */
class MLIRGenerator
{
    public:
        /// Construct a new MLIRGenerator object.
        MLIRGenerator(mlir::MLIRContext &context) : builder_(&context) {}

        /// Public API: Convert the PyBuda graph into an MLIR module operation for TTIR.
        mlir::ModuleOp emit_mlir(graphlib::Graph *graph)
        {
            graphModule_ = mlir::ModuleOp::create(get_module_location(graph), "pybuda_graph");
            graphModule_->setAttr(mlir::tt::SystemDescAttr::name,
                      mlir::tt::SystemDescAttr::getDefault(builder_.getContext()));

            builder_.setInsertionPointToStart(&graphModule_.getBodyRegion().front());

            emit_mlir_function(graph);

            /// Verify the module after we have finished constructing it, this will check
            /// the structural properties of the IR and invoke any specific verifiers we
            /// have on the TTIR operations.
            if (failed(mlir::verify(graphModule_))) 
            {
                graphModule_.emitError("module verification failed.");
                return nullptr;
            }

            mlir::OpPrintingFlags printFlags;
            printFlags.enableDebugInfo();
            graphModule_.print(llvm::outs(), printFlags);

            return graphModule_;
        }


    private:
        /// A "module" matches a PyBuda graph: containing a single function to exectue.
        mlir::ModuleOp graphModule_;
        /// The builder is a helper class to create IR. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder_;
        // The symbol table maintains a mapping between the names of pybuda nodes and their corresponding values in the current scope.
        // Initially, the function arguments (model activations) are added to the symbol table.
        // After evaluating each pybuda op node, the declare function adds a new entry to the symbol table for future reference.
        std::map<std::string, std::pair<mlir::Value, graphlib::Node*>> symbolTable_;

        /// Declares a variable in the current (only) scope.
        /// The declaration corresponds to exactly one operation node in the PyBuda graph.
        void declare(graphlib::Node *node, mlir::Value value) {
            if (symbolTable_.find(node->name()) != symbolTable_.end())
            {
                throw std::runtime_error("Variable " + node->name() + " already declared in the current scope.");
            }
            
            symbolTable_[node->name()] = {value, node};
        }

        /// Emit a new function in MLIR.
        /// A function represents a set of PyBuda operations that are executed to produce output results.
        /// This function will generate the MLIR code for each PyBuda operation in the graph and emit the return operation for the function.
        mlir::func::FuncOp emit_mlir_function(tt::graphlib::Graph *graph) {
            // Assemble the function arguments (inputs)
            llvm::SmallVector<mlir::Type> arguments;

            for (auto *input : graph->nodes_by_type(tt::graphlib::kInput))
            {
                arguments.push_back(get_node_type(input));
            }

            // Assemble the function return values (outputs)
            llvm::SmallVector<mlir::Type> returns;
            for (auto *output : graph->nodes_by_type(tt::graphlib::kOutput))
            {
                returns.push_back(get_node_type(output));
            }

            // Create the function and emit it in the MLIR module.
            auto funcType = builder_.getType<mlir::FunctionType>(mlir::TypeRange(arguments), mlir::TypeRange(returns));
            auto func = builder_.create<mlir::func::FuncOp>(graphModule_.getLoc(), "main", funcType);
            
            // Start the body of the function by creating an entry block.
            mlir::Block *entryBlock = func.addEntryBlock();

            // Declare function arguments in the symbol table
            for(auto namedValue: llvm::zip(graph->nodes_by_type(tt::graphlib::kInput), entryBlock->getArguments()))
            {
                auto node = std::get<0>(namedValue);
                auto arg = std::get<1>(namedValue);
                declare(node, arg);
            }
            
            // Set the insertion point in the builder to the beginning of the function
            // body, it will be used throughout the codegen to create operations in this
            // function.
            builder_.setInsertionPointToStart(entryBlock);

            // Walk the graph in topological order and generate MLIR for each PyBuda operation
            // node in the graph. For each new operation result, declare it in the symbol table.
            for (auto *node : graphlib::topological_sort(*graph))
            {
                // Skip if the node isn't PyBuda operation
                if (node->node_type() != tt::graphlib::NodeType::kPyOp)
                {
                    continue;
                }

                log_trace(LogMLIRGenerator, "Emitting MLIR for node {}", node->name());

                tt::graphlib::OpNode *op_node = dynamic_cast<tt::graphlib::OpNode*>(node);
                // Emit MLIR for the PyBuda operation node
                mlir::Value opValue = emit_mlir_pybuda_operation(graph, op_node);

                log_trace(LogMLIRGenerator, "Generated MLIR for node {} with value {}", node->name(), covnert_mlir_value_to_string(opValue));
            }

            emit_mlir_return_op(graph);

            return func;
        }

        /// Emit an MLIR operation for a PyBuda node.
        mlir::Value emit_mlir_pybuda_operation(tt::graphlib::Graph *graph, tt::graphlib::OpNode *op_node)
        {
            mlir::Value opResult;
            if (tt::graphlib::is_eltwise(op_node))
            {
                opResult = emit_mlir_pybuda_elementwise_op(graph, op_node);
            }

            // This is the first time we are visiting this PyBuda node during the traversal of the graph using topological sort.
            // Therefore, we need to declare the result of this operation so that we can refer to it later if needed.
            declare(op_node, opResult);

            return opResult;
        }

        /// Emit an MLIR operation for a PyBuda elementwise operation.
        mlir::Value emit_mlir_pybuda_elementwise_op(tt::graphlib::Graph *graph, tt::graphlib::OpNode *op_node)
        {
            // Evaluate operation return type
            llvm::SmallVector<mlir::Type> return_type_vector;
            return_type_vector.push_back(get_node_type(op_node));
            mlir::TypeRange return_types(return_type_vector);

            // Creating input value range for the operation
            // Since we are traversing the PyBuda graph using topological sort,
            // all operands must be present in the symbol table.
            // We iterate over the operands of the current node and retrieve their corresponding values from the symbol table.
            llvm::SmallVector<mlir::Value> input_vector;
            for (auto operand : graph->operands(op_node))
            {
                input_vector.push_back(symbolTable_.at(operand->name()).first);
            }

            mlir::ValueRange inputs(input_vector);

            // Creating output value range for the operation by creating an empty tensor to hold the output value
            llvm::SmallVector<mlir::Value> output_vector;
            output_vector.push_back(emit_mlir_empty_tensor(graph, op_node));
            mlir::ValueRange outputs = mlir::ValueRange(output_vector);

            // Create an array attribute with three elements, each representing an operand constraint of type "AnyDevice"
            auto atributes = builder_.getArrayAttr(llvm::SmallVector<mlir::Attribute>(
                3, builder_.getAttr<mlir::tt::OperandConstraintAttr>(
                           mlir::tt::OperandConstraint::AnyDevice)));

            if (op_node->op_name() == "add")
            {
                auto opResult = builder_.create<mlir::tt::ttir::AddOp>(get_pybuda_operation_location(graph, op_node), return_types, inputs, outputs, atributes);
                return opResult.getResult(0);
            }
            else if (op_node->op_name() == "multiply")
            {
                auto opResult = builder_.create<mlir::tt::ttir::MultiplyOp>(get_pybuda_operation_location(graph, op_node), return_types, inputs, outputs, atributes);
                return opResult.getResult(0);
            }
            else {
                log_error("Unsupported operation for lowering from PyBuda to TTIR: {}", op_node->op_name());
                throw std::runtime_error("Unsupported operation for lowering from PyBuda to TTIR");
            }
        }

        /// Emit an MLIR operation for an empty tensor.
        mlir::Value emit_mlir_empty_tensor(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
        {
            llvm::SmallVector<int64_t> shape_vec;
            for(auto dim : node->shape().as_vector())
            {
                shape_vec.push_back((int64_t)dim);
            }

            return builder_.create<mlir::tensor::EmptyOp>(get_pybuda_operation_location(graph, node), shape_vec, get_float_type(node));
        }

        /// Emit the return operation for the function.
        void emit_mlir_return_op(tt::graphlib::Graph *graph)
        {
            // Assemble the function return values (outputs)
            llvm::SmallVector<mlir::Value> returnValues;
            for (auto *output : graph->nodes_by_type(tt::graphlib::kOutput))
            {
                auto output_operand = graph->operands(output)[0];
                auto outputValue = symbolTable_[output_operand->name()].first;
                returnValues.push_back(outputValue);
            }

            builder_.create<mlir::func::ReturnOp>(builder_.getUnknownLoc(), mlir::ValueRange(returnValues));
        }

        /// Get the MLIR float type type for a PyBuda node.
        mlir::FloatType get_float_type(graphlib::Node *node)
        {
            switch (node->output_df())
            {
                case tt::DataFormat::Float32:
                    return builder_.getF32Type();
                case tt::DataFormat::Float16_b:
                    return builder_.getF16Type();
                case tt::DataFormat::Float16:
                    return builder_.getF16Type();
                default:
                    TT_ASSERT(false);
            }

            // TODO add all supported types in switch
            return builder_.getF32Type();
        }

        /// Get the MLIR type for a PyBuda node.
        mlir::Type get_node_type(graphlib::Node *node)
        {
            std::vector<int64_t> shape_vec;
            for (auto dim : node->shape().as_vector())
            {
                shape_vec.push_back((int64_t)dim);
            }
            return mlir::RankedTensorType::get(shape_vec, get_float_type(node));
        }

        /// Get the location for a module.
        mlir::Location get_module_location(tt::graphlib::Graph *graph)
        {
            return mlir::FileLineColLoc::get(builder_.getContext(), graph->name(), graph->id(), 0);
        }

        /// Get the simple location for a node in a format "graph_name", (graph_id), (node_id)
        mlir::Location get_node_location(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
        {
            return mlir::FileLineColLoc::get(builder_.getContext(), graph->name(), graph->id(), node->id());
        }

        /// Get the location for a PyBuda operation. The location is a combination of the operation name and the node location.
        mlir::Location get_pybuda_operation_location(tt::graphlib::Graph *graph, tt::graphlib::Node *node)
        {
            return mlir::NameLoc::get(builder_.getStringAttr(node->name()), get_node_location(graph, node));
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
};
}

namespace tt::passes
{
    /// Public API for generating MLIR from the PyBuda graph.
     mlir::OwningOpRef<mlir::ModuleOp> lower_to_mlir(graphlib::Graph * graph, mlir::MLIRContext& context)
    {
        return MLIRGenerator(context).emit_mlir(graph);
    }
}
