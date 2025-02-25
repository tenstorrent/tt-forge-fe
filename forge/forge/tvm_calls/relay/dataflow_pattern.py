from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.dataflow_pattern import ffi
from tvm.relay.expr import RelayExpr as Expr
import tvm

def construct_pre_node_map(pattern: "DFPattern", pre: Expr) -> tvm.ir.container.Map:
    """
    Construct the node_map for pre graph

    Parameters
    ----------
    pattern: tvm.relay.dataflow_pattern.DFPattern
        The pattern to match
    expr : tvm.relay.Expr
        The expression on which pattern is matched

    Returns
    -------
    result : tvm.ir.container.Map
        The dictionary constructed of pre graph subgroups in form of 
        pattern/expression pairs
    """
    return ffi.construct_pre_node_map(pattern, pre)