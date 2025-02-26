import logging
import torch
from forge.execution_tracker import ExecutionStage, record_execution_phase_and_stage

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.ir.type import TupleType
from tvm.ir.tensor_type import TensorType
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
from tvm.ir.transform import PassContext
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name, BuildModule
from tvm.ir import IRModule
from tvm.relay import function as _function
from tvm.relay.op.transform import broadcast_to
from tvm.target.compilation_config import make_compilation_config
from tvm.relay.op.contrib.register import register_pattern_table
from tvm.relay.dataflow_pattern import *
from .reportify import dump_graph
from .forge_passes import run_forge_compile_passes
from .relay_passes import run_relay_compile_passes
from .utils import *


import math
import numpy as np
from tvm.relay.dataflow_pattern import *

from loguru import logger

import networkx as nx

def _register_external_op_helper_pytorch(op_name, compiler_cfg, supported=True):
    op = tvm.ir.op.Op.get(op_name)
    if op.has_attr("target.forge_cpudevice"):
        op.reset_attr("target.forge_cpudevice")

    @tvm.ir.register_op_attr(op_name, "target.forge_cpudevice")
    def _func_wrapper(expr):
        return compiler_cfg.enable_tvm_cpu_fallback
    return _func_wrapper

def initialize_forge_cpudevice_ops(mod, compiler_cfg):
    ResetOpAttributes().visit(mod["main"])
    for op in compiler_cfg.cpu_fallback_ops:
        _register_external_op_helper_pytorch(op, compiler_cfg)
    _register_external_op_helper_pytorch("scatter_elements", compiler_cfg)

def nn_layernorm_to_forge_layernorm():
    act = wildcard()
    return is_op("layernorm")

def dense_to_matmul():
    data = wildcard()
    weight = wildcard()
    weight_t = is_op('transpose')(weight)
    return is_op('nn.dense')(data, weight_t)

def reshape_to_vstack():
    act = wildcard()
    return is_op('reshape')(act)

def reshape_to_vslice():
    act = wildcard()
    return is_op('reshape')(act)

def stack_reshape_reshape_to_binary_stack():
    act = is_tuple([wildcard(), wildcard()])
    stack = is_op("stack")(act)
    return is_op("reshape")(stack)

def concat_reshape_reshape_to_binary_stack():
    x1 = is_op("reshape")(wildcard())
    x2 = is_op("reshape")(wildcard())
    act = is_tuple([x1, x2])
    concat = is_op("concatenate")(act)
    return is_op("reshape")(concat)

def decompose_adv_index_input_tuple():
    act = wildcard()
    indices = wildcard()
    input = is_tuple([act, indices])

    return is_op("adv_index")(input)

def decompose_concatenate():
    inputs = is_tuple(None)
    return is_op("concatenate")(inputs)

def dropout_tuple_get_item():
    act = wildcard()
    dropout = is_op("nn.dropout")(act)
    return is_tuple_get_item(dropout)

def merge_conv2d_with_bias():
    input = wildcard()
    weight = wildcard()
    bias = wildcard()

    conv2d = is_op('nn.conv2d')(input, weight)
    bias_add = is_op('nn.bias_add')(conv2d, bias)

    return bias_add

def merge_conv2d_transpose_with_bias():
    input = wildcard()
    weight = wildcard()
    bias = wildcard()

    conv2d = is_op('nn.conv2d_transpose')(input, weight)
    bias_add = is_op('nn.bias_add')(conv2d, bias)

    return bias_add

def channel_last_resize():
    input = wildcard()

    transpose_input_0 = is_op("transpose")(input).has_attr({"axes": [0, 3, 2, 1]})
    transpose_input_1 = is_op("transpose")(transpose_input_0).has_attr({"axes": [0, 1, 3, 2]})

    resize = is_op("image.resize2d")(transpose_input_1).has_attr({"layout":"NCHW"})

    transpose_result_0 = is_op("transpose")(resize).has_attr({"axes": [0, 2, 1, 3]})
    transpose_result_1 = is_op("transpose")(transpose_result_0).has_attr({"axes": [0, 1, 3, 2]})

    return transpose_result_1

@register_pattern_table("forge")
def pattern_table():
    matmul = ("forge.matmul", dense_to_matmul())
    binary_stack = [
        ("forge.binary_stack", stack_reshape_reshape_to_binary_stack(), is_stack_reshape_reshape_to_binary_stack),
        ("forge.binary_stack", concat_reshape_reshape_to_binary_stack(), is_concat_reshape_reshape_to_binary_stack),
    ]
    adv_index = ("forge.adv_index", decompose_adv_index_input_tuple())
    forge_conv2d_with_bias = ("forge.forge_conv2d_with_bias", merge_conv2d_with_bias())
    forge_conv2d_transpose_with_bias = ("forge.forge_conv2d_transpose_with_bias", merge_conv2d_transpose_with_bias())
    dropout = ("forge.dropout", dropout_tuple_get_item())
    concatenate = ("forge.concatenate", decompose_concatenate())

    # channel_last_maxpool2d = ("forge.channel_last_maxpool", channel_last_maxpool())
    channel_last_resize2d = ("forge.channel_last_resize2d", channel_last_resize())
    forge_patterns = [
        *binary_stack, 
        adv_index, 
        matmul, 
        forge_conv2d_with_bias,
        forge_conv2d_transpose_with_bias,
        dropout,
        concatenate
    ]

    return forge_patterns

# TM CPU Fallback ops of interest. Ones that are valuable 
# to be included as additional fallback nodes
tm_cpu_fallback_ops_of_interest = [
    "adv_index",
    "arange",
    "broadcast_to_like",
    "broadcast_to",
    "cast_like",
    "cast",
    "collapse_sum_like",
    "collapse_sum_to",
    "concatenate",
    "contrib_reverse_reshape",
    "embedding",
    "expand_dims",
    "full_like",
    "full",
    "gather_nd",
    "gather",
    "matrix_set_diag",
    "meshgrid",
    "one_hot",
    "reinterpret",
    "repeat",
    "reshape_like",
    "reshape",
    "reverse_sequence",
    "reverse",
    "sequence_mask",
    "slice_like",
    "sparse_to_dense",
    "split",
    "squeeze",
    "stack",
    "strided_slice",
    "take",
    "tile",
    "unravel_index",
    "where",
    # Forge
    "forge.adv_index",
    "forge.binary_stack",
]

# TM CPU Fallback ops which should not be included in fallback
# itself. Those ops often are decomposed into some sort of matmuls
# which should be forced to be executed on TT device mostly.
tm_cpu_fallback_ops_to_not_include = [
    "nn.adaptive_avg_pool1d",
    "nn.adaptive_avg_pool2d",
    "nn.adaptive_avg_pool3d",
    "nn.adaptive_max_pool1d",
    "nn.adaptive_max_pool2d",
    "nn.adaptive_max_pool3d",
    "nn.avg_pool1d",
    "nn.avg_pool2d",
    "nn.avg_pool3d",
    "nn.batch_matmul",
    "nn.conv1d_transpose",
    "nn.conv1d",
    "nn.conv2d_transpose",
    "nn.conv2d",
    "nn.conv3d_transpose",
    "nn.conv3d",
    "nn.dense",
    "nn.global_avg_pool2d",
    "nn.global_max_pool2d",
    "nn.matmul",
    "nn.max_pool1d",
    "nn.max_pool2d",
    "nn.max_pool3d",
    "nn.sparse_conv2d",
    "nn.sparse_dense",
    "nn.upsampling",
    "nn.upsampling3d",
    # Forge
    "forge.forge_conv2d_transpose_with_bias",
    "forge.forge_conv2d_with_bias",
    "forge.matmul",
]

class UpdateConstants(DFPatternCallback):
    def __init__(self):
        super().__init__(rewrite_once=True)
        self.params = {}
        self.const_idx = 0
        self.pattern = is_constant()
        self.function_name = ""

    def callback(self, pre, post, node_map):
        if self.function_name not in self.params:
            self.params[self.function_name] = {}

        self.params[self.function_name][self.const_idx] = post.data
        self.const_idx += 1
        return post

class AddNopsToPassthrough(ExprMutator):
    def __init__(self):
        super().__init__()
        self.output_vars =  []

    def visit_var(self, var):
        if var in self.output_vars:
            return tvm.relay.identity(var)
        else:
            return var

    def visit_call(self, call):
        new_op = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return tvm.relay.Call(new_op, new_args, call.attrs, span=call.span)

    def visit_global_var(self, gvar):
        return gvar

    def visit_op(self, op):
        return op

    def visit_function(self, fn):
        if isinstance(fn.body, tvm.relay.expr.Tuple):
            outputs = [output for output in fn.body]
        else:
            outputs = [fn.body]

        self.output_vars.extend([output for output in outputs if isinstance(output, tvm.relay.Var)])
        new_body = self.visit(fn.body)
        return tvm.relay.Function(list(fn.params), new_body, fn.ret_type, fn.type_params, fn.attrs, span=fn.span)

def _always_true(expr):
    return True

class IdentityFunctionUnraveller(ExprMutator):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def visit_call(self, call):
        if not hasattr(call.op, 'checked_type'):
            return super().visit_call(call)

        if isinstance(call.op.checked_type, tvm.relay.FuncType):
            function = self.mod[call.op.name_hint]
            if len(function.params) == 1 and function.body == function.params[0]:
                return super().visit(call.args[0])
                
        return super().visit_call(call)

class CheckFallbackOps(ExprVisitor):
    def __init__(self, cpu_fallback_ops):
        super().__init__()
        self.cpu_fallback_ops = cpu_fallback_ops
        self.has_fallback_ops = False
        
    def visit_op(self, op): 
        if op.name in self.cpu_fallback_ops:
            self.has_fallback_ops = True 
        return super().visit_op(op)
    
   
class UnwrapForgeOpsForCPUFallback(ExprMutator):
    def __init__(self):
        super().__init__()

    def visit_call(self, call):
        if isinstance(call.op, tvm.relay.function.Function):
            if "Composite" in call.op.attrs and call.op.attrs["Composite"] == "forge_cpudevice.matmul":
                if isinstance(call.args[0], tvm.relay.expr.Call) and isinstance(call.args[0], tvm.ir.op.Op) and call.args[0].op.name == "reshape":
                    arg0 = call.args[0].args[0]
                else:
                    arg0 = call.args[0]
                if isinstance(call.args[1], (tvm.relay.expr.Var, tvm.relay.expr.Constant)):
                    arg1 = call.args[1]
                    # If transpose is part of a function, and therefore not picked up, add it explicitly
                    if isinstance(call.op.body.args[1], tvm.relay.expr.Call) and call.op.body.args[1].op.name == "transpose":
                        arg1 = tvm.relay.transpose(arg1, axes=call.op.body.args[1].attrs['axes'])
                else:
                    arg1 = call.args[1].args[0]

                logger.trace("CPU fallback on linear")
                return super().visit(tvm.relay.nn.dense(arg0, arg1))
            if "Composite" in call.op.attrs and call.op.attrs["Composite"] == "forge_cpudevice.binary_stack" and "PartitionedFromPattern" in call.op.attrs and call.op.attrs["PartitionedFromPattern"] == "Tuple_stack_reshape_":
                # Get binary_stack input
                arg0 = call.args[0] if len(call.args) > 0 else None
                arg1 = call.args[1] if len(call.args) > 1 else None
                
                # First argument is a variable or call expression
                if not (arg0 and isinstance(arg0, tvm.relay.Var) or isinstance(arg0, tvm.relay.Call)):
                    return super().visit_call(call)
                
                # Second argument isn't provided (fix for invalid binary_stack)
                if not arg1:
                    stack_attr = int(call.op.body.args[0].attrs.axis)
                    reshape_attr = [int(i) for i in call.op.body.attrs.newshape]
                    ops = tvm.relay.stack([arg0,], axis=stack_attr)
                    ops = tvm.relay.reshape(ops, newshape=reshape_attr)
                    
                    logger.info("Fixed invalid binary_stack (single input)")
                    
                    return ops

        return super().visit_call(call)


class ResetOpAttributes(ExprVisitor):
    def __init__(self):
        super().__init__()

    def visit_op(self, op):
        if op.has_attr("target.forge_cpudevice"):
            op.reset_attr("target.forge_cpudevice")
        return super().visit_op(op)

def node_hash(node):
    """Generate unique TVM node with hash and metadata.
    
    TVM node hash is unique identifier of TVM node in the specific
    graph phase. As TVM doesn't have unique node IDs this is used. 
    Also, as way of generating TVM hash can depend on op attributes 
    and position in the graph, it's common that these unique hashes 
    change with each TVM optimization pass.
    
    Besides hash, this function also populates some meta-data related
    to the node. Here is explanation regarding each of them:
    1. Node name (depends on node type, op type, etc.)
    2. Is it variable or not (params)

    Args:
        node (relay.expr.Call): Visiting node

    Returns:
        tuple: Unique node identifier
    """
    if hasattr(node, "op"):
        if isinstance(node.op, tvm.relay.function.Function):
            node_descriptor = (node.op.attrs["Composite"], False)
        elif hasattr(node.op, "name"):
            node_descriptor = (node.op.name, False)
        else:
            node_descriptor = (node.op, False)
    elif isinstance(node, tvm.relay.expr.Var):
        node_descriptor = (node.name_hint, True)
    else:
        node_descriptor = (type(node), False)

    if hasattr(node, "op") and isinstance(node.op, tvm.relay.function.Function) and node.op.id != -1:
            return (node.op.id, node_descriptor)

    if isinstance(node, tvm.relay.expr.Var) and node.id != -1:
        return (node.id, node_descriptor)
    else:
        return (tvm.ir.structural_hash(node), node_descriptor)

class NodeIndexer():
    def __init__(self):
        self.counters = ["function","call","let","var","global_var","if","tuple","tuple_getitem","constant"]
        self.increment = 100
        self.index = 0
        self.node_map = {}
    
    def reset_index(self):
        self.index = 0
    
    def count_visit(self, visitee, node=None):
        assert visitee in self.counters
        index = self.counters.index(visitee) + 1
        self.index += index * self.increment
        if node is not None:
            self.node_map[self.index] = node
        return self.index

class NodeReMapper(ExprVisitor):
    def __init__(self, node_list, node_map):
        self.node_indexer = NodeIndexer()
        self.node_map = node_map
        self.node_list = node_list
        super().__init__()

    def visit_call(self, call):
        index = self.node_indexer.count_visit("call")
        node = node_hash(call)
        if node != self.node_map[index] and self.node_map[index] in self.node_list:
            self.node_list.remove(self.node_map[index])
            self.node_list.add(node)
        return super().visit_call(call)

class ArgFallbackFinder(ExprVisitor):
    def __init__(self, graph, fallback_nodes, input_names):
        super().__init__()
        self.fallback_nodes = fallback_nodes
        self.input_names = input_names
        self.graph = graph
        
    def visit_call(self, call):
        call_hash = node_hash(call)
        if call_hash in self.fallback_nodes:
            for arg in call.args:
                arg_hash = node_hash(arg)
                if isinstance(arg, tvm.relay.Var) or arg_hash in self.fallback_nodes:
                    continue
                activation_checker = ActivationChecker(self.input_names)
                activation_checker.visit(arg)
                
                if not activation_checker.found_activation:
                    self.fallback_nodes.add(arg_hash)
                    self.fallback_nodes = self.fallback_nodes | nx.ancestors(self.graph, arg_hash )
            
            # Special case, cpu matmul with doubly-transposed weights
            if call_hash[1][0] == "forge.matmul":
                if hasattr(call.args[1], "op") and call.args[1].op.name == "transpose": # first transpose is implicit in forge.matmul
                    self.fallback_nodes.add(node_hash(call.args[1]))
            
                 
        return super().visit_call(call)
    
    def visit_tuple(self, tup):
        if node_hash(tup) in self.fallback_nodes:
            for arg in tup.fields:
                arg_hash = node_hash(arg)
                if isinstance(arg, tvm.relay.Var) or arg_hash in self.fallback_nodes:
                    continue
                activation_checker = ActivationChecker(self.input_names)
                activation_checker.visit(arg)
                
                if not activation_checker.found_activation:
                    self.fallback_nodes.add(arg_hash)
                    self.fallback_nodes = self.fallback_nodes | nx.ancestors(self.graph, arg_hash)
                  
        return super().visit_tuple(tup)

def complete_fallback_nodes(mod, graph, fallback_nodes, input_names, compiler_cfg):
    new_fallback_nodes = set()
    for node in fallback_nodes:
        new_fallback_nodes.add(node)
        ancestors = nx.ancestors(graph, node)
        descendants = nx.descendants(graph, node)
        if len(ancestors) > len(descendants):
            new_fallback_nodes = new_fallback_nodes | descendants
        else:
            if compiler_cfg.enable_tm_cpu_fallback:
                continue
            new_fallback_nodes = new_fallback_nodes | ancestors

    # Now check if we must fallback arg ancestors for fallback nodes
    arg_fallback_finder = ArgFallbackFinder(graph, new_fallback_nodes, input_names)
    arg_fallback_finder.visit(mod["main"])
    return arg_fallback_finder.fallback_nodes
    
    
class DetermineTarget(ExprMutator):
    def __init__(self, graph, fallback_nodes, compiler_cfg):
        super().__init__()
        self.users_of_unsupported_ops = 0
        self.graph = graph
        self.nodes_to_cpu_eval = set()
        self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | fallback_nodes
        self.graph_changed = True
        self.modify_graph = False
        self.compiler_cfg = compiler_cfg

    def visit_call(self, call):
        def _cpu_eval(expr):
            node = node_hash(expr)
            cpu_eval = node in self.nodes_to_cpu_eval
            if cpu_eval:
                logger.info(f"{expr.op.name} will be executed on CPU")
            return cpu_eval

        if self.modify_graph:
            if node_hash(call) in self.nodes_to_cpu_eval:
                if isinstance(call.op, tvm.relay.function.Function):
                    self.graph_changed = True
                    new_attrs = {k: (v if k != "Composite" else v.replace("forge", "forge_cpudevice")) for (k, v) in call.op.attrs.items()}
                    new_fn = call.op.with_attr(new_attrs)
                    logger.info(f"Changing {call.op.attrs['PartitionedFromPattern']}'s attr from {call.op.attrs['Composite']} to {new_fn.attrs['Composite']}")
                    return super().visit_call(tvm.relay.expr.Call(new_fn, call.args))
        elif node_hash(call) in self.nodes_to_cpu_eval and not isinstance(call.op, tvm.relay.function.Function) :
            try:
                
                tvm.ir.register_op_attr(call.op.name, "target.forge_cpudevice", _cpu_eval, level=5)
            except:
                pass

        # For non-unary ops, if one of the args is predefined for fallback and only has one output, do that op on CPU too to reduce data movement
        elif isinstance(call.op, tvm.ir.op.Op) and any([True if hasattr(arg, "op") and hasattr(arg.op, "name") and arg.op.name in self.compiler_cfg.cpu_fallback_ops else False for arg in call.args]):
            non_weight_args = [arg for arg in call.args if not isinstance(arg, tvm.relay.expr.Var)]
            if len(non_weight_args) > 1:
                call_node_ancestors = nx.ancestors(self.graph, node_hash(call))
                for arg_index, arg in enumerate(call.args):
                    output_nodes = self.graph.out_degree(node_hash(arg))
                    if isinstance(arg, tvm.relay.expr.Call) and isinstance(arg.op, tvm.ir.op.Op) and arg.op.get_attr("target.forge_cpudevice") is not None and output_nodes == 1:
                        arg_ancestors = nx.ancestors(self.graph, node_hash(arg))
                        arg_ancestors.add(node_hash(arg))
                        non_arg_ancestors = call_node_ancestors - arg_ancestors
                        contains_unsupported = any([ancestor in self.nodes_to_cpu_eval for ancestor in non_arg_ancestors])
                        if not contains_unsupported:
                            break

                        self.nodes_to_cpu_eval.add(node_hash(call))
                        self.nodes_to_cpu_eval = self.nodes_to_cpu_eval | call_node_ancestors
                        try:
                            tvm.ir.register_op_attr(call.op.name, "target.forge_cpudevice", _cpu_eval, level=5)
                        except:
                            pass
                        break

        return super().visit_call(call)


    # def visit_function(self, fn):
    #     return super().visit_function(fn)
    

def add_shared_weights_to_fallback(graph, fallback_nodes, input_names):
    added_nodes = set()
    input_nodes = [node for node in graph.nodes if node[1][1] and node[1][0] in input_names]
    for fallback_node in fallback_nodes:
        for ancestor in nx.ancestors(graph, fallback_node):
            name, maybe_param = ancestor[1]
            if not maybe_param or name in input_names:
                continue
            for output_node in graph.successors(ancestor):
                # if the output node or any of its discendants is not the fallback node
                if output_node != fallback_node and fallback_node not in nx.descendants(graph, output_node):
                    index = 0
                    nodes_to_check = [output_node]
                    while index < len(nodes_to_check):
                        node = nodes_to_check[index]
                        added_nodes.add(node)
                        if any(an for an in nx.ancestors(graph, node) if an in input_nodes):
                            break
                        nodes_to_check.extend(graph.successors(node))
                        index += 1

    return added_nodes | fallback_nodes


def extend_fallback_with_tm_ops(graph, fallback_nodes, max_depth, ops_of_interest, ops_to_avoid):
    logger.trace("Checking for fallback nodes based on perf...")
    
    # Gether all DiGraph output nodes (includes subgraphs too)
    output_nodes = [u for u, deg in graph.out_degree() if not deg]
    
    if len(output_nodes) != 1:
        # Subgraphs are often TVM functions represented as standalone graph, so we'll ignore them
        # as they are referenced in the original DiGraph
        logger.warning("Extended TM CPU Fallback: Multiple output nodes in DiGraph, using one with most ancestor nodes.")

    # Use graph which has most nodes (highly probability to be considered)
    # as main (most of subgraphs are wrapped TVM Functions with couple of ops)
    max_ancestors = 0
    output_node = None
    for out_n in output_nodes:
        ancestors = nx.ancestors(graph, out_n)
        if max_ancestors < len(ancestors):
            max_ancestors = len(ancestors)
            output_node = out_n
            
    # Initialize DiGraph attributes
    for node in graph.nodes():
        graph.nodes[node]['exec_on_cpu'] = False
        graph.nodes[node]['path_suitable_for_fallback'] = True
            
    # Traverse all paths until certain depth and mark valid candidates
    # as CPU fallback nodes
    tm_fallback_traverse(graph, output_node, max_depth, ops_of_interest, ops_to_avoid, 0)
    
    # Optional for debugging purposes
    print_extended_tm_fallback_graph = False
    if print_extended_tm_fallback_graph:
        # Color DiGraph for easier debugging. Meaning of each color:
        # - green - op of interest, but not on suitable fallback path
        # - blue - op is on suitable fallback path, but doesn't contain op of interest on path
        # - red - op is not suitable fallback path as it contains invalid op as one of its descendants
        # - purple - op is suitable for CPU execution as its on valid path and has ops of interests as descendants
        for node in graph.nodes():
            if graph.nodes[node]['exec_on_cpu']:
                graph.nodes[node]['color'] = 'green'
                
            if graph.nodes[node]['path_suitable_for_fallback']:
                graph.nodes[node]['color'] = 'blue'
            else:
                graph.nodes[node]['color'] = 'red'
                
            if graph.nodes[node]['exec_on_cpu'] and graph.nodes[node]['path_suitable_for_fallback']:
                graph.nodes[node]['color'] = 'purple'
        
        # Visualize DiGraph
        # 
        # Useful online visualizer: https://dreampuf.github.io/GraphvizOnline
        dot_graph = nx.nx_pydot.to_pydot(graph)
        print(dot_graph)
    
    # Gather additional CPU fallback nodes
    additional_fallback_ops = set()
    for node in graph.nodes():
        if not (graph.nodes[node]['exec_on_cpu'] and graph.nodes[node]['path_suitable_for_fallback']):
            continue
        
        op_name = str(node[1][0])
        logger.trace("Additional descendant for CPU: {}".format(op_name))
        
        additional_fallback_ops.add(node)
        
    return additional_fallback_ops | fallback_nodes

def tm_fallback_traverse(graph, current_node, max_depth, ops_of_interest, ops_to_avoid, current_depth):
    # Traversal depth exit condition
    if current_depth >= max_depth:
        logger.trace("Stopping traverse for given path as max allowed depth is reached")
        return
    current_depth += 1

    # Invalid paths exit condition
    for node, descendants in nx.bfs_successors(graph, current_node):
        valid_descendants_fallback_path = True
        if not descendants:
            continue

        for descendant in descendants:
            valid_descendants_fallback_path &= graph.nodes[descendant]["path_suitable_for_fallback"]
            
        if not valid_descendants_fallback_path:
            logger.trace("Stopping traverse for given path all descendant paths are invalid for fallback")
            return

    logger.trace("Current: {}".format(current_node[1][0]))
    predecessors = graph.predecessors(current_node)
    for predecessor in predecessors:
        node_desc = predecessor[1]
        op = node_desc[0]
        op_name = str(op)
        logger.trace("Predecessor: {}".format(op_name))
            
        # Handle ops to avoid
        if op_name in ops_to_avoid:
            graph.nodes[predecessor]["path_suitable_for_fallback"] &= False
        graph.nodes[predecessor]["path_suitable_for_fallback"] &= graph.nodes[current_node]["path_suitable_for_fallback"]
        
        # Handle conv based ops to avoid
        if op_name in ops_to_avoid and "conv" in op_name and "bias" not in op_name:
            graph.nodes[current_node]["exec_on_cpu"] = False
            graph.nodes[current_node]["path_suitable_for_fallback"] &= False
        
        # Handle ops of interest
        if op_name in ops_of_interest:
            graph.nodes[predecessor]["exec_on_cpu"] = True
            
            # Handle descendants if path is suitable for fallback
            for node, children in nx.bfs_successors(graph, predecessor):
                logger.trace("Correcting descendants for: {}".format(node))
                for child in children:
                    logger.trace("Descendant: {}".format(child))

                    if graph.nodes[child]['exec_on_cpu']:
                        logger.trace("Child ({}) already executes on CPU, breaking further traverse".format(child))
                        break

                    if graph.nodes[child]['path_suitable_for_fallback']:
                        logger.trace("Child ({}) is corrected to be executed on CPU".format(child))
                        graph.nodes[child]['exec_on_cpu'] = True
                    
        tm_fallback_traverse(graph, predecessor, max_depth, ops_of_interest, ops_to_avoid, current_depth)
        
    logger.trace("Finishing traverse for given path on depth: {}".format(current_depth))

class IndexChecker(ExprVisitor):
    
    def __init__(self):
        super().__init__()
        self.all_nodes_non_float = True
        
    def visit(self, node):
        if hasattr(node, "checked_type"):
            if "float" in node.checked_type.dtype:
                self.all_nodes_non_float = False
                return
        super().visit(node)
        
class ActivationChecker(ExprVisitor):
    def __init__(self, input_names):
        super().__init__()
        self.input_names = input_names
        self.found_activation = False
        
    def visit_var(self, var):
        if var.name_hint in self.input_names:
            self.found_activation = True
        super().visit_var(var)

class EnumerateNodes(ExprMutator):
    def __init__(self):
        super().__init__()
        self.index = 0
        self.var_to_index = {}

    def unique_index(self):
        self.index +=1
        return self.index
    
    def visit_function(self, fn):
        new_params = [self.visit(x) for x in fn.params]
        new_body = self.visit(fn.body)
        return tvm.relay.function.FunctionWithFields(fn, list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs, id=self.unique_index())

    def visit_call(self, call):
        new_op =self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return tvm.relay.Call(new_op, new_args, call.attrs, call.type_args, call.span, id=self.unique_index())
        
    def visit_var(self, var):
        if var.name_hint in self.var_to_index:
            id = self.var_to_index[var.name_hint]
        else:
            id = self.unique_index()
            self.var_to_index[var.name_hint] = id

        return tvm.relay.Var(var.name_hint, type_annotation=var.type_annotation, framework_dtype=var.framework_dtype, span=var.span, id=id)

    def visit_tuple_getitem(self, op):
        new_tuple_value = self.visit(op.tuple_value)
        return tvm.relay.TupleGetItem(new_tuple_value, op.index, id=self.unique_index(), span=op.span)

    def visit_tuple(self, tup):
        new_fields = [self.visit(field) for field in tup.fields]

        return tvm.relay.Tuple(new_fields, tup.span, id=self.unique_index())

    def visit_constant(self, const):
        return tvm.relay.Constant(const.data, const.is_param, const.name, const.framework_dtype, const.span, id=self.unique_index())

class HashGraph(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.calls_nodes = set()
        
    def visit_call(self, call):
        self.calls_nodes.add((call, node_hash(call)))
        super().visit_call(call)

class ConstructDiGraph(ExprVisitor):
    
    def __init__(self):
        super().__init__()
        self.graph = nx.MultiDiGraph()
        self.fallback_nodes = set()
        self.node_indexer = NodeIndexer()
        self.names_used = {}

    def register_args(self, parent, parent_node):
        for arg in parent.args:
            if isinstance(arg, tvm.relay.expr.Var):
                pass
    
            self.graph.add_edge(node_hash(arg), parent_node)

    def visit_call(self, call):
        node = node_hash(call)
        self.node_indexer.count_visit("call", node)
        if isinstance(call.op, tvm.ir.op.Op) and call.op.get_attr("target.forge_cpudevice") is not None:
            self.fallback_nodes.add(node)
            logger.info(f"Adding: {call.op} to fallback")
        elif (
            isinstance(call.op, tvm.relay.function.Function) 
            and isinstance(call.op.body, tvm.relay.expr.TupleGetItem)
            and call.op.body.tuple_value.op.get_attr("target.forge_cpudevice") is not None
        ):
            self.fallback_nodes.add(node)
            logger.info(f"Adding: {call.op.body.tuple_value.op} to fallback")

        elif (
            isinstance(call.op, tvm.relay.function.Function) 
            and not isinstance(call.op.body, tvm.relay.expr.TupleGetItem)
            and call.op.body.op.get_attr("target.forge_cpudevice") is not None
        ):
            self.fallback_nodes.add(node)
            logger.info(f"Adding: {call.op.body.op} to fallback")
            if node[1][0] == "forge.adv_index":
                logger.trace("Special case: adv_index. If none of the ancestors of the indices are float, fallback all ancestors to indices")
                index_checker = IndexChecker()
                index_checker.visit(call.args[1])
                if index_checker.all_nodes_non_float:
                    logger.info("All ancestors of indices are non-float, fallback all ancestors to indices")
                    hash_graph = HashGraph()
                    hash_graph.visit(call.args[1])
                    for pair in hash_graph.calls_nodes:
                        logger.info("Adding: adv_index to fallback")
                        self.fallback_nodes.add(pair[1])
                        self.register_args(pair[0], pair[1])
                
        self.register_args(call, node)
        # Make sure CPU output shape starts with 1
        if (
            isinstance(call.op, tvm.ir.op.Op) 
            and call.op.name == "reshape"
            and call.checked_type.shape[0] == 1
            and isinstance(call.args[0], tvm.relay.expr.Call)
            and isinstance(call.args[0].op, tvm.ir.op.Op)
            and call.args[0].op.name == "embedding"
            and call.args[0].op.get_attr("target.forge_cpudevice") is not None
        ):
            self.fallback_nodes.add(node)
            logger.info(f"Adding: {call.op} to fallback")

        return super().visit_call(call)
    
    def visit_tuple_getitem(self, t):
        node = node_hash(t)
        self.register_args(t.tuple_value, node)
        return super().visit_tuple_getitem(t)

    def visit_tuple(self, t):
        tuple_node = node_hash(t)
        for producer in t.fields:
            producer_node = node_hash(producer)
            self.graph.add_edge(producer_node, tuple_node)
        return super().visit_tuple(t)
    


class MainFunctionFinder(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.funcs = []
        self.cpu_funcs = []
        self.tt_funcs = []

    def visit_call(self, call):
        if isinstance(call.op, tvm.relay.expr.GlobalVar):
            if "forge_cpudevice_main" in call.op.name_hint:
                self.funcs.append(call.op)
                self.cpu_funcs.append(call.op)
            elif "forge_main" in call.op.name_hint:
                self.funcs.append(call.op)
                self.tt_funcs.append(call.op)
        super().visit_call(call)
            

class NodeOriginFinder(ExprVisitor):
    def __init__(self, origins, include_constants=True):
        super().__init__()
        self.produced_by = {}
        self.consumed_by = {}
        self.origin = None
        self.possible_origins = origins
        self.include_constants_as_origin = include_constants
    
    def reset(self):
        self.origin = None

    def visit_call(self, call):
        
        if call.op in self.possible_origins:
            self.origin = (call.op.name_hint, 0)
            return
        super().visit_call(call)

    def visit_constant(self, const):
        if self.include_constants_as_origin:
            self.origin = ('constant', -1)

    def visit_var(self, var):
        if var in self.possible_origins:
            self.origin = var

    def visit_tuple_getitem(self, t):
        if isinstance(t.tuple_value, tvm.relay.Call):
            if t.tuple_value.op in self.possible_origins:
                self.origin = (t.tuple_value.op.name_hint, t.index)
                return

        super().visit_tuple_getitem(t)
        
    def visit_global_var(self, gvar):
        if gvar in self.possible_origins:
            self.origin = gvar

def trace_to_origin(node, possible_origins, include_constants=True):
    pd = NodeOriginFinder(possible_origins, include_constants)
    pd.visit(node)
    return pd.origin

class PartitionFinder(ExprVisitor):
    def __init__(self, mod, tt_funcs, cpu_funcs):
        super().__init__()
        self.tt_funcs = tt_funcs
        self.cpu_funcs = cpu_funcs
        self.mod = mod
        self.cpu_pre_funcs = []
        self.cpu_post_funcs = []
        
    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.expr.GlobalVar):
            gvar = call.op
            if gvar in self.cpu_funcs:
                # determine if its pre or post
                origin = trace_to_origin(call, self.tt_funcs)
                logger.trace(f"{gvar} will be executed on CPU")
                if origin and origin[0] in [func.name_hint for func in self.tt_funcs]:
                    self.cpu_post_funcs.append(gvar)
                    
                    # Given that this cpu function has an input which originates from a tt function,
                    # we want to make sure that no outputs of this function are consumed by a tt function.
                    for func in self.tt_funcs:
                        tt_func_callnodes = extract_function_callnodes(self.mod["main"], [func])
                        assert len(tt_func_callnodes) == 1, "No tt function should be called more than once."
                        assert not trace_to_origin(tt_func_callnodes[0], [gvar]), "There is a CPU function which has inputs originating from a TT function, and outputs consumed by a TT function. This should not happen and is not supported."
                    
                else:
                    self.cpu_pre_funcs.append(gvar)
        super().visit_call(call)
                

def get_relay_output(mod, params, inputs, target):
    # Build and Run Relay modules with inputs as (key : tensor) pair
    # Then, inputs dont need to be in the same order as 'mod' defines.
    ret_type = mod["main"].checked_type.ret_type
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build_module.build(mod, target=target, params=params)
    m = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    m.run(**inputs)
    
    def _unflatten(flat_iter, cur_type):
        import tvm.relay.ty as _ty
        if isinstance(cur_type, _ty.TensorType):
            return next(flat_iter)
        if isinstance(cur_type, _ty.TupleType):
            fields = []
            for field_type in cur_type.fields:
                field = _unflatten(flat_iter, field_type)
                fields.append(field)
            return fields
        raise ValueError("Return type", ret_type, "contains unsupported type", cur_type)

    flattened = []
    import tvm.runtime.ndarray as _nd
    for i in range(m.get_num_outputs()):
        flattened.append(m.get_output(i).copyto(_nd.cpu(0)))
    relay_outputs = _unflatten(iter(flattened), ret_type)

    if not isinstance(relay_outputs, (list, tuple)):
        relay_outputs = [relay_outputs]
    relay_outputs = [x.numpy() for x in flattened]
    return relay_outputs


def verify_outputs(framework_outputs, relay_outputs, compile_location, rtol=1e-02, atol=1e-04, pcc=None):
    allowed_to_fail = False
    if len(framework_outputs) != len(relay_outputs):
        logger.error(f"Different number of outputs. Framework: {len(framework_outputs)}, TVM: {len(relay_outputs)} after {compile_location}")

    for i, (fr_out, tvm_out) in enumerate(zip(framework_outputs, relay_outputs)):
        if fr_out.shape != tvm_out.shape:
            logger.error(f"Different shapes for outputs. Framework: {fr_out.shape}, TVM: {tvm_out.shape} after {compile_location}")

        if pcc is None:
            ok = np.allclose(fr_out, tvm_out, rtol=rtol, atol=atol, equal_nan=True)
        else:
            pcc_value = np.min(np.ma.corrcoef(np.ma.masked_invalid(fr_out.flatten()), np.ma.masked_invalid(tvm_out.flatten())))
            if isinstance(pcc_value, np.ma.core.MaskedConstant):
                pcc_value = 1.0
            ok = pcc_value >= pcc

        if not ok:
            logger.error(f"Tensor mismatch on output {i} between framework and TVM after {compile_location}.")
            logger.trace(f"Framework: (shape = {fr_out.shape}")
            logger.trace(fr_out)
            logger.trace(f"TVM: (shape = {tvm_out.shape}")
            logger.trace(tvm_out)
            logger.info("Max ATOL Delta: " + "{:.3e}".format(np.max(np.abs((fr_out - tvm_out))).item()) + ", atol=" +  "{}".format(atol))
            logger.info("Max RTOL Delta: " + "{:.3e}".format(np.max(np.abs((fr_out - tvm_out))/tvm_out).item()) + ", rtol=" + "{}".format(rtol))
            if pcc is not None:
                logger.info(f"PCC got={pcc_value}, required={pcc}")
            if not allowed_to_fail:
                raise RuntimeError

    logger.info(f"Verified TVM Relay outputs against framework outputs after {compile_location}")

def verify_tvm_compile(mod, params, inputs, target, framework_outputs, compile_location, verify_cfg=None):
    relay_outputs = get_relay_output(mod, params, inputs, target)

    # Verify compile passes (original relay passes + forge passes)
    if verify_cfg:
        verify_outputs(framework_outputs, relay_outputs, compile_location, rtol=verify_cfg.rtol, atol=verify_cfg.atol, pcc=verify_cfg.pcc)
    else:
        verify_outputs(framework_outputs, relay_outputs, compile_location)


class CompareWarner(DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True, rewrite_once=True)
        
        self.act1 = wildcard()
        self.act2 = wildcard()
        self.pattern = is_op("equal")(self.act1, self.act2) \
                        | is_op("not_equal")(self.act1, self.act2) \
                        | is_op("less")(self.act1, self.act2) \
                        | is_op("less_equal")(self.act1, self.act2) \
                        | is_op("greater")(self.act1, self.act2) \
                        | is_op("greater_equal")(self.act1, self.act2)

    def callback(self, pre, post, node_map):
        pre_node_map = construct_pre_node_map(self.pattern, pre)
        op_name = pre_node_map[self.pattern][0].op.name
        act1 = pre_node_map[self.act1][0]
        act2 = pre_node_map[self.act2][0]
        if "int" in act1.checked_type.dtype or "int" in act2.checked_type.dtype:
            logger.warning(f"Integer input(s) detected in comparison op: {op_name}. This may cause data mismatch.")
        return post
    
def warn_of_int_comparisons(mod):
    warner = CompareWarner()
    rewrite(warner, mod['main'])
    
def compile_for_forge(relay_module, graph_name, target='llvm', params=None, inputs=None, framework_outputs=None, verify_cfg=None):

    if not isinstance(relay_module, (IRModule, _function.Function)):
        raise ValueError("Type of input parameter mod must be tvm.IRModule")

    if isinstance(relay_module, _function.Function):
        if params:
            relay_module = bind_params_by_name(relay_module, params)
        relay_module = IRModule.from_expr(relay_module)
        logger.warning(
            "Please use input parameter mod (tvm.IRModule) "
            "instead of deprecated parameter func (tvm.relay.function.Function)"
        )

    tophub_context = tvm.autotvm.utils.EmptyContext()

    with tophub_context, tvm.transform.PassContext(opt_level=5):
        logger.trace("Before Compiling")
        logger.trace(relay_module.functions)
        dump_graph(relay_module, graph_name, "before_compiling")

        relay_module = run_relay_compile_passes(relay_module)
        dump_graph(relay_module, graph_name, "after_relay_passes")
        record_execution_phase_and_stage(ExecutionStage.TVM_RELAY_IR_TRANSFORM)

        compiled_relay_module = run_forge_compile_passes(relay_module, params, inputs, target, framework_outputs, verify_cfg)
        dump_graph(compiled_relay_module, graph_name, "after_forge_passes")
        record_execution_phase_and_stage(ExecutionStage.TVM_PATTERN_CALLBACKS)
        
        # Integer comparisons may lead to incorrect results on HW
        warn_of_int_comparisons(compiled_relay_module)

    return compiled_relay_module, params

class FlattenInputs(ExprMutator):

    def __init__(self, flattenend_name_map):
        super().__init__()
        self.flattened_name_map = flattenend_name_map
        self.input_tuples= []
        self.new_params = []

    def visit_tuple_getitem(self, op):
        if op.tuple_value in self.input_tuples:
            tup_index = self.tuple_indices[self.input_tuples.index(op.tuple_value)]
            return self.visit(self.old_param_map[tup_index][op.index])
        
        new_op = tvm.relay.TupleGetItem(self.visit(op.tuple_value), op.index, span=op.span)
        return new_op

    def visit_call(self, call):
        new_op = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return tvm.relay.Call(new_op, new_args, call.attrs, span=call.span)

    def visit_function(self, fn):
        new_body = fn.body
        self.old_param_map = {}
        self.tuple_indices = []

        param_num = 0

        for i in range(len(fn.params)):
            if isinstance(fn.params[i].type_annotation, TupleType):
                self.input_tuples.append(fn.params[i])
                inputs = fn.params[i].type_annotation.fields
                new_params = []
                
                for j in range(len(inputs)):
                    input = inputs[j]
                    new_params.append(tvm.relay.Var(self.flattened_name_map[fn.params[i].name_hint][j], input))
                    param_num += 1

                    if i not in self.old_param_map:
                        self.tuple_indices.append(i)
                        self.old_param_map[i] = {}
                    self.old_param_map[i][j] = new_params[-1]
   
                self.new_params += new_params
                
            else:
                self.new_params.append(fn.params[i])
                self.old_param_map[i] = self.new_params[-1]
                

        new_body = self.visit(fn.body)
        return tvm.relay.Function(self.new_params, new_body, fn.ret_type, fn.type_params, fn.attrs, span=fn.span)
    
class FlattenOutputs(ExprMutator):

    def visit_tuple_getitem(self, op):
        # Forego the need to have a TupleGetItem node and just get the item
        return op.tuple_value[op.index]
    
    def visit_function(self, fn):
        new_body = fn.body
        def tuple_in_tuple(tup):
            for field in tup.fields:
                if isinstance(field, tvm.relay.expr.Tuple):
                    return field
            return False
        
        if isinstance(new_body, tvm.relay.expr.Tuple):
            tup = tuple_in_tuple(new_body)
            while tup:
                new_fields = []
                for field in new_body.fields:
                    if field == tup:
                        new_fields.extend([self.visit(field) for field in tup.fields])
                    else:
                        new_fields.append(field)
                new_body = tvm.relay.expr.Tuple(new_fields)
                tup = tuple_in_tuple(new_body)
        
        return tvm.relay.Function(fn.params, new_body, fn.ret_type, fn.type_params, fn.attrs, span=fn.span)

def flatten_IO(mod, flattened_name_map):

    # flattens inputs in IR
    mod["main"] = FlattenInputs(flattened_name_map).visit(mod["main"])
    logger.trace("After FlattenInputs")
    logger.trace(mod.functions)
    
    mod["main"] = FlattenOutputs().visit(mod["main"])
    logger.trace("After FlattenOutputs")
    logger.trace(mod.functions)
    return mod


def fallback_on_cpu(mod, compiler_cfg, input_names):
    import time
    
    logger.trace(f"Running cpu fallback compilation")
    logger.trace(f"Checking if the graph has any cpu-fallback ops...")
    start = time.time() 
    check_fallback_ops = CheckFallbackOps(compiler_cfg.cpu_fallback_ops)
    check_fallback_ops.visit(mod["main"])
    logger.trace(f"Done, took: {(time.time() - start):.2f} s")
    
    if check_fallback_ops.has_fallback_ops or compiler_cfg.enable_tm_cpu_fallback:
        logger.trace(f"Constructing digraph...")
        start = time.time()
        graph_constructor = ConstructDiGraph()
        graph_constructor.visit(mod["main"])
        logger.trace(f"Done, took: {(time.time() - start):.2f} s")
        
        # Visualize DiGraph
        #
        # Useful online visualizer: https://dreampuf.github.io/GraphvizOnline
        # dot_graph = nx.nx_pydot.to_pydot(graph_constructor.graph)
        # print(dot_graph)

        logger.trace(f"Finding and adding shared weights...")
        start = time.time()
        fallback_nodes = add_shared_weights_to_fallback(graph_constructor.graph, graph_constructor.fallback_nodes, input_names)
        
        # Extend fallback with valid TM ops from the end of the graph
        if compiler_cfg.enable_tm_cpu_fallback:
            fallback_nodes = extend_fallback_with_tm_ops(
                graph_constructor.graph, fallback_nodes,
                compiler_cfg.tm_cpu_fallback_max_depth,
                tm_cpu_fallback_ops_of_interest,
                tm_cpu_fallback_ops_to_not_include)
        
        logger.trace(f"Done, took: {(time.time() - start):.2f} s")
        logger.trace(f"Determining target for ops...")
        
        fallback_nodes = complete_fallback_nodes(mod, graph_constructor.graph, fallback_nodes, input_names, compiler_cfg)
        
        terget_determiner = DetermineTarget(graph_constructor.graph, fallback_nodes, compiler_cfg)
        new_mod = None
        start = time.time()
        terget_determiner.modify_graph = False
        mod["main"] = terget_determiner.visit(mod["main"])
        terget_determiner.memo_map = {}
        terget_determiner.modify_graph = True
        mod["main"] = terget_determiner.visit(mod["main"])
        logger.trace(f"Done, took: {(time.time() - start):.2f} s")
        logger.trace(f"Remapping nodes...")

        start = time.time()
        node_remapper = NodeReMapper(terget_determiner.nodes_to_cpu_eval, graph_constructor.node_indexer.node_map)
        node_remapper.visit(mod["main"])
        terget_determiner.nodes_to_cpu_eval = node_remapper.node_list
        logger.trace("After DetermineTarget")
        logger.trace(mod.functions)
        logger.trace(f"Done, took: {(time.time() - start):.2f} s")


class VarConverter(ExprMutator):
    def __init__(self, key, replacement):
        super().__init__()
        self.key = key
        self.replacement = replacement
        
    def visit_call(self, call):
        new_args = []
        for arg in call.args:
            if isinstance(arg, tvm.relay.expr.Var) and arg.name_hint == self.key.name_hint:
                new_args.append(self.replacement)
            else:
                new_args.append(arg)
                
        
        return super().visit_call(tvm.relay.expr.Call(call.op, new_args, call.attrs, call.type_args))

class FunctionPlacer(ExprMutator):
    def __init__(self, mod, output_map, input_map, new_func_gvar, new_args):
        super().__init__()
        self.output_map = output_map
        self.input_map = input_map
        self.new_func_gvar = new_func_gvar
        self.mod = mod
        self.new_args = new_args
        self.inserted_node = None
    
    def visit_tuple_getitem(self, tgi):
        if isinstance(tgi.tuple_value, tvm.relay.expr.Call):
            call = tgi.tuple_value
            if isinstance(call.op, tvm.relay.expr.GlobalVar) and call.op.name_hint in self.output_map:
                if not self.inserted_node:
                    self.inserted_node = tvm.relay.expr.Call(self.new_func_gvar, [])
                    
                new_index = self.output_map[call.op.name_hint][tgi.index]
                return tvm.relay.expr.TupleGetItem(self.inserted_node, new_index)
            
        return super().visit_tuple_getitem(tgi)
    
    def visit_call(self, call):
        if isinstance(call.op, tvm.relay.expr.GlobalVar) and call.op.name_hint in self.output_map: 
            if not self.inserted_node:
                self.inserted_node = tvm.relay.expr.Call(self.new_func_gvar, [], call.attrs, call.type_args)
                
                if isinstance(self.mod[self.new_func_gvar].checked_type.ret_type, tvm.ir.type.TupleType):
                    getitem = tvm.relay.expr.TupleGetItem(self.inserted_node, self.output_map[call.op.name_hint][0])
                    return getitem
                else:
                    return self.inserted_node
            else:
                if isinstance(self.mod[self.new_func_gvar].checked_type.ret_type, tvm.ir.type.TupleType):
                    getitem = tvm.relay.expr.TupleGetItem(self.inserted_node, self.output_map[call.op.name_hint][0])
                    return getitem
                else:
                    return self.inserted_node
            
        return super().visit_call(call)

class FunctionCallNodeFinder(ExprVisitor):

    def __init__(self, function_names):
        super().__init__()
        self.func_names = function_names
        self.funcs = []

    def visit_call(self, call):
        if isinstance(call.op, tvm.relay.GlobalVar):
            if call.op.name_hint in self.func_names:
                self.funcs.append(call)
        return super().visit_call(call)

def extract_function_callnodes(main_module, funcs):
    ff = FunctionCallNodeFinder([func.name_hint for func in funcs])

    ff.visit(main_module)
    return ff.funcs

class FunctionArgPlacer(ExprMutator):
    
    def __init__(self, func, new_args):
        super().__init__()
        assert isinstance(func, tvm.ir.GlobalVar), "Must supply function GlobalVar"
        
        self.func = func
        self.new_args = new_args
        
    def visit_call(self, call):
        if isinstance(call.op, tvm.relay.expr.GlobalVar) and call.op.name_hint == self.func.name_hint:
            new_fn = self.visit(call.op)
            self.new_args = [self.visit(arg) for arg in self.new_args]
            return tvm.relay.expr.Call(new_fn, self.new_args, call.attrs, call.type_args, call.span)
        return super().visit_call(call)

def merge_functions(mod, funcs_to_merge, new_name):
    assert all([isinstance(func, tvm.relay.expr.GlobalVar) for func in funcs_to_merge]), "Must supply functions as GlobalVars"
    funcs = []
    new_output_map = {}
    new_input_map = {}
    
    new_params = []
    new_type_params = []
    new_body = []
    
    if len(funcs_to_merge) <= 1:
        return mod
    
    for i, func in enumerate(funcs_to_merge):
        new_output_map[func.name_hint] = {}
        body = mod[func].body

        # Merge bodies
        if isinstance(body, tvm.relay.expr.Tuple):
            for j, output in enumerate(body.fields):
                new_output_map[func.name_hint][j] = len(new_body)
                new_body.append(output)
        else:
            new_output_map[func.name_hint][0] = len(new_body)
            new_body.append(body)
            
    if len(new_body) == 1:
        new_body = new_body[0]
    else:
        new_body = tvm.relay.expr.Tuple(new_body)
        
    for i, func in enumerate(funcs_to_merge):
        params = mod[func].params
        type_params = mod[func].type_params
        
        new_input_map[func.name_hint] = {}
        
        # Merge params/type params
        for j, param in enumerate(params):
            new_param_names = [p.name_hint for p in new_params]
            if not param.name_hint in new_param_names:
                new_input_map[func.name_hint][j] = len(new_params)
                new_params.append(param)
            else:
                # convert all usages of this param to the param currently in the list
                for name in new_param_names:
                    if name == param.name_hint:
                        new_input_map[func.name_hint][j] = new_param_names.index(name)
                        new_body = VarConverter(param, new_params[new_param_names.index(name)]).visit(new_body)
                        
                
        for type_param in type_params:
            if not type_param in new_type_params:
                new_type_params.append(type_param)
    
    new_fn = tvm.relay.Function(new_params, new_body)
    new_attrs = {k: (v if k != "Composite" else v.replace("forge", "forge_cpudevice")) for (k, v) in mod[funcs_to_merge[0]].attrs.items()}
    new_attrs["global_symbol"] = new_name
    new_fn = new_fn.with_attr(new_attrs)
    
    gvar = tvm.ir.expr.GlobalVar(new_name)
    mod.update_func(gvar, new_fn)
        
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    logger.trace("After InferType")
    logger.trace(mod.functions)
    
    # Replace functions with merged function
    fn_placer = FunctionPlacer(mod, new_output_map, new_input_map, gvar, new_params)
    placed = fn_placer.visit(mod["main"])
    
    # Place arguments for new function in correct order
    new_args = []
    
    while len(new_args) < len(new_params):
        for fn_name, input_map in new_input_map.items():
            fn_gvar = mod.get_global_vars()[[gvar.name_hint for gvar in mod.get_global_vars()].index(fn_name)]
            fn_callnode = extract_function_callnodes(mod["main"], [fn_gvar])[0]
            for old_idx, new_idx in input_map.items():
                if new_idx == len(new_args):
                    arg = fn_callnode.args[old_idx]
                    if isinstance(arg, tvm.relay.expr.Call) and isinstance(arg.op, tvm.relay.expr.GlobalVar):
                        callnodes = extract_function_callnodes(placed, [arg.op])
                        if len(callnodes):
                            arg = callnodes[0]
                    if isinstance(arg, tvm.relay.expr.TupleGetItem):
                        if isinstance(arg.tuple_value, tvm.relay.expr.Call) and isinstance(arg.tuple_value.op, tvm.relay.expr.GlobalVar):
                            callnodes = extract_function_callnodes(placed, [arg.tuple_value.op])
                            if len(callnodes):
                                arg = tvm.relay.TupleGetItem(callnodes[0], arg.index, span=callnodes[0].span)
                    new_args.append(arg)
                    
    placed = FunctionArgPlacer(gvar, new_args).visit(placed)
    mod["main"] = placed
    
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    logger.trace("After InferType")
    logger.trace(mod.functions)
    
    return mod

class TupleGetItemIndexSwapper(ExprMutator):
    def __init__(self, relavent_tup, old_idx, new_idx):
        super().__init__()
        self.relevant_tup = relavent_tup
        self.old_idx = old_idx
        self.new_idx = new_idx
        self.already_changed = []
        
    def visit_tuple_getitem(self, tup_getitem):
        if tup_getitem.tuple_value == self.relevant_tup and tup_getitem not in self.already_changed:
            if tup_getitem.index == self.old_idx:
                self.already_changed.append(super().visit_tuple_getitem(tvm.relay.expr.TupleGetItem(tup_getitem.tuple_value, self.new_idx)))
                return self.already_changed[-1]
            elif tup_getitem.index == self.new_idx:
                self.already_changed.append(super().visit_tuple_getitem(tvm.relay.expr.TupleGetItem(tup_getitem.tuple_value, self.old_idx)))
                return self.already_changed[-1]
            # elif self.old_idx < self.new_idx and self.new_idx >= tup_getitem.index > self.old_idx:
            #     self.already_changed.append(super().visit_tuple_getitem(tvm.relay.expr.TupleGetItem(tup_getitem.tuple_value, tup_getitem.index - 1)))
            #     return self.already_changed[-1]
            
            # elif self.old_idx > self.new_idx and self.new_idx <= tup_getitem.index < self.old_idx:
            #     self.already_changed.append(super().visit_tuple_getitem(tvm.relay.expr.TupleGetItem(tup_getitem.tuple_value, tup_getitem.index + 1)))
            #     return self.already_changed[-1]
        return super().visit_tuple_getitem(tup_getitem)

class CallNodeReplacer(ExprMutator):
    def __init__(self, old_callnode, replacement):
        super().__init__()
        self.old_callnode = old_callnode
        self.replacement = replacement
        
    def visit_call(self, call):
        if call == self.old_callnode:
            if isinstance(self.replacement, tvm.relay.expr.Call):
                self.replacement = super().visit_call(self.replacement)
                return self.replacement
            else:
                self.replacement = super().visit(self.replacement)
                return self.replacement
        return super().visit_call(call)

class TupleGetItemReplacer(ExprMutator):
    def __init__(self, old_tup_getitem, new_tup_getitem):
        super().__init__()
        self.old_tup_getitem = old_tup_getitem
        self.new_tup_getitem = new_tup_getitem
        self.replaced = False
        
    def visit_tuple_getitem(self, tup_getitem):
        if tup_getitem == self.old_tup_getitem:
            if isinstance(self.new_tup_getitem, tvm.relay.expr.TupleGetItem):
                return super().visit_tuple_getitem(self.new_tup_getitem)
            else:
                return super().visit(self.new_tup_getitem)
        return super().visit_tuple_getitem(tup_getitem)
        
    def visit_call(self, call):
        if call == self.old_tup_getitem:
            if isinstance(self.new_tup_getitem, tvm.relay.expr.TupleGetItem):
                return super().visit_tuple_getitem(self.new_tup_getitem)
            else:
                return super().visit(self.new_tup_getitem)
        return super().visit_call(call)

def align_func_io(mod, cpu_pre_func, tt_func, cpu_post_func):

    def swap_outputs(mod, func, old_idx, new_idx):
        
        function = mod[func]
        new_body = function.body

        assert isinstance(new_body, tvm.relay.expr.Tuple), "Expected tuple output"
        outputs = list(new_body.fields)
        if new_idx >= len(outputs):
            new_idx = len(outputs) - 1
        
        tmp = outputs[old_idx]
        outputs[old_idx] = outputs[new_idx]
        outputs[new_idx] = tmp
        new_body = tvm.relay.expr.Tuple(outputs)
        
        new_return_type = list(function.ret_type.fields)
        tmp = new_return_type[old_idx]
        new_return_type[old_idx] = new_return_type[new_idx]
        new_return_type[new_idx] = tmp
        new_fn = tvm.relay.Function(function.params, new_body, ret_type=tvm.relay.TupleType(new_return_type), attrs=function.attrs, span=func.span)
        mod[func] = new_fn

        # swap tuplegetitem indices in main module
        func_callnode = extract_function_callnodes(mod["main"], [func])[0]
        mod["main"] = TupleGetItemIndexSwapper(func_callnode, old_idx, new_idx).visit(mod["main"])
        return mod

    def swap_inputs(mod, func, old_idx, new_idx):
        function = mod[func]

        # Switch params
        new_params = list(function.params)
        tmp = new_params[old_idx]
        new_params[old_idx] = new_params[new_idx]
        new_params[new_idx] = tmp
        
        new_fn = tvm.relay.Function(new_params, function.body, ret_type=function.ret_type, attrs=function.attrs, span=func.span)
        mod[func] = new_fn
        
        # switch args in main module
        func_callnode = extract_function_callnodes(mod["main"], [func])[0]
        new_args = list(func_callnode.args)
        tmp = new_args[old_idx]
        new_args[old_idx] = new_args[new_idx]
        new_args[new_idx] = tmp
        new_call = tvm.relay.expr.Call(func_callnode.op, new_args, attrs=func_callnode.attrs)
        
        mod["main"] = FunctionArgPlacer(func, new_args).visit(mod["main"])#CallNodeReplacer(func_callnode, new_call).visit(mod["main"])
        
        return mod
    
    def permute_list(l, permute_map):
        assert len(l) == len(permute_map.items()), "Require an i->j mapping for each element in the list"
        
        new_list = [None] * len(l)
        for old_idx, new_idx in permute_map.items():
            new_list[new_idx] = l[old_idx]

        return new_list
    # if there is a cpu pre function, align its output with the tt function input
    if cpu_pre_func:
        tt_func_callnode = extract_function_callnodes(mod["main"], [tt_func])[0]
        tt_arg_origins = [trace_to_origin(arg, [cpu_pre_func]) for arg in tt_func_callnode.args]
        
        inter_fn_idx = 0
        for input_idx, origin in enumerate(tt_arg_origins):
            if origin is not None:
                if inter_fn_idx != input_idx:
                    assert input_idx > inter_fn_idx, "Expected input_idx to be greater than inter_fn_idx"

                    # This means that the current tt func arg originates from the cpu pre func, but weights come before it, move to the front
                    mod = swap_inputs(mod, tt_func, input_idx, inter_fn_idx)
                inter_fn_idx += 1
        
        # Retrieve these two again since the order may have changed
        tt_func_callnode = extract_function_callnodes(mod["main"], [tt_func])[0]
        tt_arg_origins = [trace_to_origin(arg, [cpu_pre_func]) for arg in tt_func_callnode.args]

        for input_idx, origin in enumerate(tt_arg_origins):
            if origin is not None:
                _, output_idx = origin
                if input_idx != output_idx and isinstance(mod[cpu_pre_func].body, tvm.relay.expr.Tuple):
                    mod = swap_outputs(mod, cpu_pre_func, output_idx, input_idx)

        # Find a map for which outputs from cpu pre go to which inputs of tt
        tt_func_callnode = extract_function_callnodes(mod["main"], [tt_func])[0]
        tt_arg_origins = [trace_to_origin(arg, [cpu_pre_func]) for arg in tt_func_callnode.args]
        io_map = {}
        num_args = 0
        for input_idx, origin in enumerate(tt_arg_origins):
            if origin is not None:
                _, output_idx = origin
                io_map[input_idx] = output_idx
                num_args += 1

        # Permute the params of the function
        tt_function = mod[tt_func]
        new_params = list(tt_function.params)
        new_params = permute_list(new_params[:num_args], io_map) + new_params[num_args:] # Everything after index: num_args are actual parameters, not activations
        new_tt_function = tvm.relay.Function(new_params, tt_function.body, ret_type=tt_function.ret_type, attrs=tt_function.attrs, span=tt_function.span)
        mod[tt_func] = new_tt_function
        
        # Permute the args of the tt function call
        tt_func_callnode = extract_function_callnodes(mod["main"], [tt_func])[0]
        old_args = list(tt_func_callnode.args)[:num_args] 
        new_args = permute_list(old_args, io_map) + list(tt_func_callnode.args)[num_args:] # Everything after index: num_args are actual parameters, not activations
        new_call = tvm.relay.expr.Call(tt_func_callnode.op, new_args, attrs=tt_func_callnode.attrs)
        mod['main'] = FunctionArgPlacer(tt_func, new_args).visit(mod["main"])
        
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    
    # if there is a cpu post function, align its input with the tt function output
    if cpu_post_func:
        cpu_post_func_callnode = extract_function_callnodes(mod["main"], [cpu_post_func])[0]
        cpu_post_arg_origins = [trace_to_origin(arg, [tt_func]) for arg in cpu_post_func_callnode.args]
        
        inter_fn_idx = 0
        for input_idx, origin in enumerate(cpu_post_arg_origins):
            if origin is not None:
                if inter_fn_idx != input_idx:
                    assert input_idx > inter_fn_idx, "Expected input_idx to be greater than inter_fn_idx"
                    
                    # This means that the current tt func arg originates from the cpu pre func, but weights come before it, move to the front
                    mod = swap_inputs(mod, cpu_post_func, input_idx, inter_fn_idx)
                inter_fn_idx += 1
        
        # Retrieve these two again since the order may have changed
        cpu_post_func_callnode = extract_function_callnodes(mod["main"], [cpu_post_func])[0]
        cpu_post_arg_origins = [trace_to_origin(arg, [tt_func]) for arg in cpu_post_func_callnode.args]
        for input_idx, origin in enumerate(cpu_post_arg_origins):
            if origin is not None:
                _, output_idx = origin
                if input_idx != output_idx and isinstance(mod[tt_func].body, tvm.relay.expr.Tuple):
                    # swap the outputs of tt_func
                    mod = swap_outputs(mod, tt_func, output_idx, input_idx)

        # Find a map for which outputs from tt go to which inputs of cpu post
        cpu_post_func_callnode = extract_function_callnodes(mod["main"], [cpu_post_func])[0]
        cpu_post_arg_origins = [trace_to_origin(arg, [tt_func]) for arg in cpu_post_func_callnode.args]
        io_map = {}
        num_args = 0
        for input_idx, origin in enumerate(cpu_post_arg_origins):
            if origin is not None:
                _, output_idx = origin
                io_map[input_idx] = output_idx
                num_args += 1

        # Permute the params of the function
        cpu_post_function = mod[cpu_post_func]
        new_params = list(cpu_post_function.params)
        new_params = permute_list(new_params[:num_args], io_map) + new_params[num_args:] # Everything after index: num_args are actual parameters, not activations
        new_cpu_post_function = tvm.relay.Function(new_params, cpu_post_function.body, ret_type=cpu_post_function.ret_type, attrs=cpu_post_function.attrs, span=cpu_post_function.span)
        mod[cpu_post_func] = new_cpu_post_function
        
        # Permute the args of the tt function call
        cpu_post_func_callnode = extract_function_callnodes(mod["main"], [cpu_post_func])[0]
        old_args = list(cpu_post_func_callnode.args)[:num_args]
        new_args = permute_list(old_args, io_map) + list(cpu_post_func_callnode.args)[num_args:] # Everything after index: num_args are actual parameters, not activations
        new_call = tvm.relay.expr.Call(cpu_post_func_callnode.op, new_args, attrs=cpu_post_func_callnode.attrs)
        mod['main'] = FunctionArgPlacer(cpu_post_func, new_args).visit(mod["main"])
        
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    return mod

def add_passthrough_variable(mod, func, output_node, var_name):
    
    # add new param
    new_params = list(mod[func].params)
    func_callnode = extract_function_callnodes(mod["main"], [func])[0]
    new_var = None
    existing_var = None
    # Do not add param if it already exists, only need to pass it through as output as well
    if output_node not in func_callnode.args:
        new_var = tvm.relay.var(var_name, output_node.checked_type)
        new_params.append(new_var)
    else:
        existing_var = mod[func].params[list(func_callnode.args).index(output_node)]
        
    output_op = None
    if isinstance(output_node, tvm.relay.expr.TupleGetItem):
        output_op = output_node.tuple_value.op
    elif isinstance(output_node, tvm.relay.expr.Call):
        output_op = output_node.op
    elif isinstance(output_node, tvm.relay.expr.Var):
        pass
    else:
        assert False, f"Cannot handle output node of type {type(output_node)}" 
    
    # add new output
    orig_output = mod[func].body
    new_num_outputs = 0
    if not isinstance(orig_output, tvm.relay.expr.Tuple) and new_var:
        output = [orig_output, new_var]
        new_num_outputs = 2
    elif isinstance(orig_output, tvm.relay.expr.Tuple) and new_var:
        output = [*list(orig_output.fields), new_var]
        new_num_outputs = len(output)
    elif isinstance(orig_output, tvm.relay.expr.Tuple) and not new_var:
        output = [*list(orig_output.fields), existing_var]
        new_num_outputs = len(output)
    else:
        output = [orig_output, existing_var]
        new_num_outputs = 2
    
    
    new_body = tvm.relay.expr.Tuple(output)
    new_return_type = []
    if not isinstance(mod[func].ret_type, tvm.relay.ty.TupleType):
        new_return_type = [mod[func].ret_type]
    else:
        new_return_type.extend(mod[func].ret_type.fields)
    new_return_type.append(output_node.checked_type)
    new_fn = tvm.relay.Function(new_params, new_body, ret_type=tvm.relay.TupleType(new_return_type), attrs=mod[func].attrs, span=mod[func].span)

    mod[func] = new_fn
    
    # Fix first output retrieval if func output was not originally a tuple
    if not isinstance(orig_output, tvm.relay.expr.Tuple):
        func_callnode = extract_function_callnodes(mod["main"], [func])[0]
        # Must clone or CallNodeReplacer will recurse infinitely
        func_callnode_clone = tvm.relay.expr.Call(func_callnode.op, func_callnode.args, attrs=func_callnode.attrs)
        tgi = tvm.relay.expr.TupleGetItem(func_callnode_clone, 0)
        mod["main"] = CallNodeReplacer(func_callnode, tgi).visit(mod["main"])
    
    # Add output var to args in passthrough in main
    func_callnode = extract_function_callnodes(mod["main"], [func])[0]
    
    if new_var:
        if isinstance(output_node, tvm.relay.expr.TupleGetItem):
            originating_func_callnode = extract_function_callnodes(mod["main"], [output_op])[0]
            passthrough_node = tvm.relay.expr.TupleGetItem(originating_func_callnode, output_node.index)
        elif isinstance(output_node, tvm.relay.expr.Var):
            passthrough_node = output_node
        else:
            originating_func_callnode = extract_function_callnodes(mod["main"], [output_op])[0]
            passthrough_node = originating_func_callnode
        
        new_args = [*list(func_callnode.args), passthrough_node]
        new_call = tvm.relay.expr.Call(func_callnode.op, new_args, attrs=func_callnode.attrs)
        replacer = CallNodeReplacer(func_callnode, new_call)
        mod["main"] = replacer.visit(mod["main"])
        
        # Replace output tuplegetitem with new tuplegetitem
        new_output_tgi = tvm.relay.expr.TupleGetItem(replacer.replacement, new_num_outputs - 1)
    else:
        # Must add new tuplegetitem and replace output with that, rather than swapping out the existing one
        new_output_tgi = tvm.relay.expr.TupleGetItem(func_callnode, new_num_outputs - 1)

    return mod, new_output_tgi
    
def handle_output_passthrough(mod, cpu_pre_func, tt_func, cpu_post_func):
    
    output = mod["main"].body
    
    func_order = [func for func in [cpu_pre_func, tt_func, cpu_post_func] if func is not None]
    if not isinstance(output, tvm.relay.expr.Tuple):
        # Single output must come from last function by this point.
        assert trace_to_origin(output, [func_order[-1]]), "Expected output to come from last function"
        return mod
    
    origins = [param for param in mod["main"].params] + [tt_func]
    if cpu_pre_func:
        origins.append(cpu_pre_func)
    if cpu_post_func:
        origins.append(cpu_post_func)
        
    outputs = list(output.fields)
    
    output_origins = [(output, trace_to_origin(output, origins)) for output in outputs]
    passthrough_count = 0
    for out_idx, (out, output_origin) in enumerate(output_origins):
        if output_origin is not None:
            if output_origin[0] != func_order[-1].name_hint:
        
                originating_func_idx = 0
                for func in func_order:
                    if func.name_hint == output_origin[0]:
                        break
                    originating_func_idx += 1
                
                originating_func = func_order[originating_func_idx]
                passthrough_funcs = func_order[originating_func_idx+1:]
                
                
                for passthrough_func in passthrough_funcs:
                    # type is required for add_passthrough_variable
                    mod = tvm.transform.Sequential([transform.InferType()])(mod)
                    output_node = mod["main"].body.fields[out_idx]
                    mod, new_output_node = add_passthrough_variable(mod, passthrough_func, output_node, f"passthrough_{passthrough_count}")
                    
                    # Passthrough to output
                    passthrough_func_callnode = extract_function_callnodes(mod["main"], [passthrough_func])[0]
                    output_node = mod["main"].body.fields[out_idx]
                    if output_node not in passthrough_func_callnode.args:
                        # Re-retrieve output since its predecessors may have changed
                        mod["main"] = TupleGetItemReplacer(output_node, new_output_node).visit(mod["main"])
                    else:
                        outputs = list(mod["main"].body.fields)
                        outputs[out_idx] = new_output_node
                        new_main_fn = tvm.relay.Function(mod["main"].params, tvm.relay.expr.Tuple(outputs), ret_type=mod["main"].ret_type, attrs=mod["main"].attrs, span=mod["main"].span)
                        mod["main"] = new_main_fn
                    
                    passthrough_count += 1
       
    return mod
                
def handle_input_passthrough(mod, cpu_pre_func, tt_func, cpu_post_func, input_names):
    output = mod["main"].body
    model_params = [param for param in mod["main"].params if param.name_hint in input_names]
    func_order = [func for func in [cpu_pre_func, tt_func, cpu_post_func] if func is not None]
    if len(func_order) == 1:
        return mod
    
    for param in model_params:
        funcs_that_use_param = []
        param_idx_map ={}
        for func in func_order:
            func_callnode = extract_function_callnodes(mod["main"], [func])[0]
            if param in func_callnode.args:
                funcs_that_use_param.append(func)
                param_idx_map[func] = list(func_callnode.args).index(param)
        
        # Need to pass through all funcs that use param before funcs_that_use_param[-1]
        
        passthrough_funcs = func_order[:func_order.index(funcs_that_use_param[-1])]

        old_func_arg = param
        for passthrough_func in passthrough_funcs:
            # type is required for add_passthrough_variable
            mod = tvm.transform.Sequential([transform.InferType()])(mod)
            mod, new_func_arg = add_passthrough_variable(mod, passthrough_func, old_func_arg, param.name_hint)
            
            # convert all args that use old_func_arg to use new_func_arg
            funcs_to_change_arg = func_order[func_order.index(passthrough_func)+1:]
            
            for func_to_change_arg in funcs_to_change_arg:
                if func_to_change_arg in param_idx_map:
                    func_callnode = extract_function_callnodes(mod["main"], [func_to_change_arg])[0]
                    old_func_arg = func_callnode.args[param_idx_map[func_to_change_arg]]
                    new_args = list(func_callnode.args)
                    new_args[param_idx_map[func_to_change_arg]] = new_func_arg
                    new_call = tvm.relay.Call(func_callnode.op, new_args, func_callnode.attrs, span=func_callnode.span)
                    mod["main"] = FunctionArgPlacer(func_to_change_arg, new_args).visit(mod["main"])
                    
                    # Retrieve new func_callnode
                    mod = tvm.transform.Sequential([transform.InferType()])(mod)
                    func_callnode = extract_function_callnodes(mod["main"], [func_to_change_arg])[0]
                    new_func_arg = func_callnode.args[param_idx_map[func_to_change_arg]]
                    
            old_func_arg = new_func_arg
    return mod

def handle_inter_func_passthrough(mod, cpu_pre_func, tt_func, cpu_post_func):
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    # Both cpu pre and post must exist for this to be necesarry
    if not (cpu_pre_func and cpu_post_func):
        return mod
    
    # Find args of cpu post that are directly produced in cpu pre
    cpu_post_callnode = extract_function_callnodes(mod["main"], [cpu_post_func])[0]
    
    cpu_post_args_from_pre = [trace_to_origin(arg, [cpu_pre_func]) for arg in cpu_post_callnode.args]
    cpu_post_args_from_tt = [trace_to_origin(arg, [tt_func]) for arg in cpu_post_callnode.args]
    
    # trace_to_origin will trace right through the tt_func and return the cpu pre output if it exists
    # So, we need to check both lists.
    # If an element of cpu_post_args_from_tt is None, then the element at the same index of cpu_post_args_from_pre 
    # is the cpu pre output that is directly passed to cpu post
    
    passthrough_args = []
    cpu_post_arg_indices_to_change = []
    for idx, (from_pre, from_tt) in enumerate(zip(cpu_post_args_from_pre, cpu_post_args_from_tt)):
        if from_tt is None and from_pre is not None:
            passthrough_args.append(from_pre)
            cpu_post_arg_indices_to_change.append(idx)
    
    # If there are no passthrough args found then we dont need to do anything
    if len(passthrough_args) == 0:
        return mod
    
    # At this point, each element in passthrough_args is an output of cpu pre that is directly passed to cpu post
    tt_func_callnode = extract_function_callnodes(mod["main"], [tt_func])[0]
    cpu_pre_callnode = extract_function_callnodes(mod["main"], [cpu_pre_func])[0]
    
    new_tt_args = list(tt_func_callnode.args)
    
    for arg in passthrough_args:
        if isinstance(cpu_pre_callnode.checked_type, tvm.ir.type.TupleType):
            _, output_idx = arg
            new_tt_args.append(tvm.relay.TupleGetItem(cpu_pre_callnode, output_idx, span=cpu_pre_callnode.span))
        else:
            assert len(passthrough_args) == 1, "More than one passthrough arg, but cpu pre output is not a tuple"
            new_tt_args.append(cpu_pre_callnode)

    old_tt_func_num_outputs = len(mod[tt_func].body.fields) if isinstance(mod[tt_func].body, tvm.relay.expr.Tuple) else 1
    new_tt_body = mod[tt_func].body
    new_tt_params = []
    for i, arg in enumerate(passthrough_args):
        if isinstance(cpu_pre_callnode.checked_type, tvm.ir.type.TupleType):
            _, output_idx = arg
            output_node = tvm.relay.TupleGetItem(cpu_pre_callnode, output_idx, span=cpu_pre_callnode.span)
            new_tt_params.append(tvm.relay.Var(f"inter_cpu_passthrough_{i}", cpu_pre_callnode.checked_type.fields[output_idx]))
        else:
            new_tt_params.append(tvm.relay.Var(f"inter_cpu_passthrough_{i}", cpu_pre_callnode.checked_type))
        
        
    if not isinstance(new_tt_body, tvm.relay.expr.Tuple):
        new_tt_body = tvm.relay.Tuple([new_tt_body] + new_tt_params)
    else:
        new_tt_body = tvm.relay.Tuple(list(new_tt_body.fields) + new_tt_params)
    
    new_tt_return_type = []
    if not isinstance(mod[tt_func].ret_type, tvm.relay.ty.TupleType):
        new_tt_return_type = [mod[tt_func].ret_type]
    else:
        new_tt_return_type.extend(mod[tt_func].ret_type.fields)
    new_tt_return_type.extend([new_tt_params[i].type_annotation for i in range(len(new_tt_params))])

    mod[tt_func] = tvm.relay.Function(list(mod[tt_func].params) + new_tt_params, new_tt_body, ret_type=tvm.relay.TupleType(new_tt_return_type), attrs=mod[tt_func].attrs, span=mod[tt_func].span)
    # Fix first output retrieval if func output was not originally a tuple
    if not isinstance(mod[tt_func].body, tvm.relay.expr.Tuple):
        tt_func_callnode = extract_function_callnodes(mod["main"], [func])[0]
        # Must clone or CallNodeReplacer will recurse infinitely
        tt_func_callnode_clone = tvm.relay.expr.Call(tt_func_callnode.op, tt_func_callnode.args, attrs=tt_func_callnode.attrs)
        tgi = tvm.relay.expr.TupleGetItem(func_callnode_clone, 0)
        mod["main"] = CallNodeReplacer(func_callnode, tgi).visit(mod["main"])
    
    mod["main"] = FunctionArgPlacer(tt_func, new_tt_args).visit(mod["main"])
    tt_func_callnode = extract_function_callnodes(mod["main"], [tt_func])[0]
    cpu_post_callnode = extract_function_callnodes(mod["main"], [cpu_post_func])[0]
    new_cpu_post_args = list(cpu_post_callnode.args)
    for i, idx in enumerate(cpu_post_arg_indices_to_change):
        new_cpu_post_args[idx] = tvm.relay.TupleGetItem(tt_func_callnode, old_tt_func_num_outputs + i, span=tt_func_callnode.span)

    mod["main"] = FunctionArgPlacer(cpu_post_func, new_cpu_post_args).visit(mod["main"])
    mod = tvm.transform.Sequential([transform.InferType()])(mod)
    return mod

def partition_for_forge(mod, graph_name, compiler_cfg, input_names=[]):
    initialize_forge_cpudevice_ops(mod, compiler_cfg)

    with tvm.transform.PassContext(opt_level=5):
        logger.trace("partition_for_forge:: At Entry")
        logger.trace(mod.functions)
        
        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        logger.trace("After InferType")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.MergeComposite(pattern_table())])(mod)
        logger.trace("After MergeComposite")
        logger.trace(mod.functions)

        mod["main"] = AddNopsToPassthrough().visit(mod["main"])
        logger.trace("After AddNopsToPassthrough")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        logger.trace("After InferType")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.FoldConstant()])(mod)
        logger.trace("After FoldConstant")
        logger.trace(mod.functions)

        mod["main"] = EnumerateNodes().visit(mod["main"])
        logger.trace("After EnumerateNodes")
        logger.trace(mod.functions)
        
        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        logger.trace("After InferType")
        logger.trace(mod.functions)

        if compiler_cfg.enable_tvm_cpu_fallback:
            fallback_on_cpu(mod, compiler_cfg, input_names)

        mod = tvm.transform.Sequential([transform.AnnotateTarget(["forge_cpudevice", "forge"])])(mod)
        logger.trace("After AnnotateTarget")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.MergeCompilerRegions()])(mod)
        logger.trace("After MergeCompilerRegions")
        logger.trace(mod.functions)

        mod = tvm.transform.Sequential([transform.PartitionGraph(bind_constants=True)])(mod)
        logger.trace("After PartitionGraph")
        logger.trace(mod.functions)

        for k, v in mod.global_var_map_.items():
            if "cpudevice" in k:
                mod[v] = UnwrapForgeOpsForCPUFallback().visit(mod[v])
        logger.trace("After UnwrapForgeOpsForCPUFallback")
        logger.trace(mod.functions)

        # Unravel f(x) = x functions
        # Note that this applies only to tvm function calls where the input is returned as the output.
        # This will not change any identity op calls
        mod["main"] = IdentityFunctionUnraveller(mod).visit(mod["main"])
        logger.trace("After IdentityFunctionUnraveller")
        logger.trace(mod.functions)
        
        func_finder = MainFunctionFinder()
        func_finder.visit(mod["main"])

        partition_finder = PartitionFinder(mod, func_finder.tt_funcs, func_finder.cpu_funcs)
        partition_finder.visit(mod["main"])

        if len(partition_finder.cpu_pre_funcs) > 0:
            logger.info("A CPU pre-process device has been created")
        if len(partition_finder.cpu_post_funcs) > 0:
            logger.info("A CPU post-process device has been created")

        # now we have the pre/post functions figured out
        # merge the pre-functions, tt-functions, and post-functions into one each
        #     i.e we have one cpu pre function, one cpu post function, and one tt function
        mod = merge_functions(mod, partition_finder.cpu_pre_funcs, "tvmgen_default_forge_cpudevice_main_pre")
        mod = merge_functions(mod, partition_finder.tt_funcs, "tvmgen_default_forge_main")
        mod = merge_functions(mod, partition_finder.cpu_post_funcs, "tvmgen_default_forge_cpudevice_main_post")
        
        # Assert that merge_functions merges cpu pre/post into at most one each, and that there is exactly one tt function
        func_finder = MainFunctionFinder()
        func_finder.visit(mod["main"])

        partition_finder = PartitionFinder(mod, func_finder.tt_funcs, func_finder.cpu_funcs)
        partition_finder.visit(mod["main"])
        assert len(partition_finder.cpu_pre_funcs) <= 1, "There should only be one cpu pre function"
        assert len(partition_finder.tt_funcs) == 1, "There should only be one tt function"
        assert len(partition_finder.cpu_post_funcs) <= 1, "There should only be one cpu post function"
        
        # Handle passthrough
        # All outputs of mod["main"] should come from the last function (tt_func or cpu_post_func if it exists)
        # Any outputs that come before this should be passed through each descendant function in an f(x) = x manner
        cpu_pre_func = partition_finder.cpu_pre_funcs[0] if len(partition_finder.cpu_pre_funcs) > 0 else None
        tt_func = partition_finder.tt_funcs[0]
        cpu_post_func = partition_finder.cpu_post_funcs[0] if len(partition_finder.cpu_post_funcs) > 0 else None

        # This handles the cases where a model input is directly consumed in the cpu post func (if it exists) or tt func (if the cpu pre exists)
        mod = handle_input_passthrough(mod, cpu_pre_func, tt_func, cpu_post_func, input_names)
        # This handles the case where an output produced by CPU pre is consumed by CPU post
        mod = handle_inter_func_passthrough(mod, cpu_pre_func, tt_func, cpu_post_func)
        
        # This handles the cases where the tt func or even cpu pre func produces a model output.
        # We do this after aligning so that in the generated python code all of the unused passthrough 
        # variables come after all of the variables consumed by the module.
        mod = handle_output_passthrough(mod, cpu_pre_func, tt_func, cpu_post_func)
        
        # Since the tvm graph is valid, the order of the outputs/inputs to each function doesnt matter yet.
        # However, if for example the 10th output of the tt function is the 2nd input of the cpu post function,
        # the json graphs that get generated will not contain the information about the order of the inputs/outputs.
        mod = align_func_io(mod, cpu_pre_func, tt_func, cpu_post_func)

        if not isinstance(mod["main"].body, tvm.relay.expr.Tuple):
            main_body_call_node = [mod["main"].body]
        else:
            main_body_call_node = mod["main"].body

        unsupported_op_names = []
        for item in main_body_call_node:
            if isinstance(item, tvm.relay.expr.Call):
                for arg in item.args:
                    if isinstance(arg, tvm.relay.expr.Var):
                        continue
                    if isinstance(arg, tvm.relay.expr.Tuple):
                        continue
                    if isinstance(arg, tvm.relay.expr.TupleGetItem):
                        continue
                    # if isinstance(arg, tvm.relay.expr.TupleGetItem):
                    #     # arg = arg.tuple_value().op
                    #     continue
                    
                    if not isinstance(arg.op, tvm.ir.expr.GlobalVar):
                        unsupported_op_names.append(arg.op.name)

                    assert arg.op in mod.global_var_map_.values(), mod["main"]
        
        if len(unsupported_op_names) > 0:
            print("Operators: " + str(unsupported_op_names) + " are unsupported.")
            assert False

        assert len(mod.global_var_map_) > 1, f"No forge compatible graph can be generated"

        constant_updator = UpdateConstants()

        for i in range(len(mod.global_var_map_)):
            function_name = mod.get_global_vars()[i].name_hint
            ResetOpAttributes().visit(mod[mod.get_global_vars()[i]])
            if function_name == "main":
                continue
            constant_updator.function_name = function_name
            rewrite(constant_updator, mod[mod.get_global_vars()[i]])
        params = constant_updator.params
        
    # Convert NaN attributes to Zeros
    for partition_key, partition_val in params.items():
        for param_key, param_val in params[partition_key].items():
            param_val_np = param_val.asnumpy()
            if np.isnan(param_val_np).all():
                zero_mtx = tvm.nd.array(np.zeros(param_val_np.shape).astype(param_val.dtype))
                params[partition_key][param_key] = zero_mtx

    dump_graph(mod, graph_name, "after_forge_partition")
        
    return mod, params
