# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import json

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprVisitor, ExprMutator


def get_default_reportify_path(test_name):
    home = os.environ["HOME"]
    return home + "/testify/ll-sw/" + test_name


def initialize_directory(path, test_name):
    filename = path + "/" + "summary.yaml"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"content:\n")
        f.write(f"  name: ll-sw.{test_name}\n")
        f.write(f"  output-dir: {path}\n")
        f.write(f"type: summary\n")


def get_tvm_reports_relpath():
    return "/tvm_reports/Passes/"


class CreateJson(ExprVisitor):
    def __init__(self):
        super().__init__()
        self.node_idx = 0
        self.graph = {}
        self.graph["graph"] = {}
        self.graph["nodes"] = {}

        self.node_map = {}
        self.visited_tgvs = {}

    def get_default_op(self, name):
        op = {}
        op["name"] = name
        op["epoch"] = 0
        op["ir"] = "forge"
        op["unique_id"] = self.node_idx
        op["input_nodes"] = []
        op["output_nodes"] = []
        op["forge"] = 1
        self.graph["nodes"][name] = op
        self.node_idx += 1
        return op

    def visit_call(self, call):
        if call not in self.visited_tgvs:
            if hasattr(call.op, "name"):
                name_partial = call.op.name
            elif hasattr(call.op, "name_hint"):
                name_partial = call.op.name_hint
            else:
                name_partial = call.op.attrs["Composite"]  # Composite functions
            name = f"{name_partial}_{self.node_idx}"
            op = self.get_default_op(name)
            if isinstance(call.checked_type, tvm.ir.type.TupleType):
                shape = []
                for field in call.checked_type.fields:
                    if hasattr(field, "shape"):
                        shape.append([int(dim) for dim in field.shape])
                op["cache"] = {"shape": shape}
            else:
                op["cache"] = {
                    "shape": [
                        int(dim) if isinstance(dim, tvm.tir.expr.IntImm) else -1 for dim in call.checked_type.shape
                    ]
                }
            op["class"] = name_partial
            op["type"] = name_partial
            op["opcode"] = "RelayOp"
            if hasattr(call, "span"):
                op["span"] = repr(call.span)
            self.node_map[call] = name
        return super().visit_call(call)

    def visit_var(self, call):
        name = call.name_hint
        op = self.get_default_op(name)

        if type(call.checked_type) == tvm.ir.type.TupleType:
            shape = []
            for field in call.checked_type.fields:
                if hasattr(field, "shape"):
                    shape.append([int(dim) for dim in field.shape])
            op["cache"] = {"shape": shape}
        else:
            op["cache"] = {
                "shape": [
                    int(dim) if isinstance(dim, (int, float, complex)) else str(dim) for dim in call.checked_type.shape
                ]
            }
        op["opcode"] = "Input"
        op["class"] = "Input::"
        op["type"] = "Input::input"
        self.node_map[call] = name
        return super().visit_var(call)

    def visit_constant(self, const):
        name = f"constant_{self.node_idx}"
        op = self.get_default_op(name)
        op["cahce"] = {"shape": [int(dim) for dim in const.checked_type.shape]}
        op["opcode"] = "Input"
        op["class"] = "Input::"
        op["type"] = "Input::constant"
        self.node_map[const] = name
        return super().visit_constant(const)

    def visit_tuple_getitem(self, t):
        if t.tuple_value in self.visited_tgvs:
            self.node_map[t] = self.visited_tgvs[t.tuple_value]
        else:
            if isinstance(t.tuple_value, tvm.relay.expr.Tuple):
                op_type = "tuple"
            elif hasattr(t.tuple_value.op, "name"):
                op_type = t.tuple_value.op.name
            elif hasattr(t.tuple_value.op, "name_hint"):
                op_type = t.tuple_value.op.name_hint
            else:
                op_type = "Unknown"
            name = f"{op_type}_{self.node_idx}"
            op = self.get_default_op(name)
            if hasattr(t.checked_type, "shape"):
                assert not any(
                    isinstance(dim, tvm.tir.expr.Any) for dim in t.checked_type.shape
                ), "Dynamic shapes not supported"
                op["cache"] = {"shape": [int(dim) for dim in t.checked_type.shape]}
            else:
                op["cache"] = {"shape": []}
            op["class"] = op_type
            op["type"] = op_type
            op["opcode"] = "RelayOp"
            self.visited_tgvs[t.tuple_value] = name
            self.node_map[t] = name
        return super().visit_tuple_getitem(t)

    def visit_tuple(self, tup):
        name = f"tuple_{self.node_idx}"
        op = self.get_default_op(name)
        shape = []
        for field in tup.checked_type.fields:
            if hasattr(field, "shape"):
                shape.append([int(dim) if isinstance(dim, (int, float, complex)) else str(dim) for dim in field.shape])

        op["cache"] = {"shape": shape}
        op["class"] = "tuple"
        op["type"] = "tuple"
        op["opcode"] = "RelayOp"
        self.node_map[tup] = name
        return super().visit_tuple(tup)


def convert_serialized_tvm_to_reportify_graph(mod):
    json_creator = CreateJson()
    json_creator.visit(mod)

    graph = json_creator.graph

    def link_tuple(node, name):
        for field in node.fields:
            if isinstance(field, tvm.relay.expr.Tuple):
                link_tuple(field, field)
                continue
            else:
                graph["nodes"][name]["input_nodes"].append(json_creator.node_map[field])
                graph["nodes"][json_creator.node_map[field]]["output_nodes"].append(name)

    # because expr visitor does not operate in topologial order, link all
    # the nodes after they are visited
    for node, name in json_creator.node_map.items():
        if isinstance(node, tvm.relay.expr.Tuple):
            link_tuple(node, name)
        if isinstance(node, tvm.relay.expr.TupleGetItem):
            if isinstance(node.tuple_value, tvm.relay.expr.Tuple):
                link_tuple(node.tuple_value, name)
                continue
            args = node.tuple_value.args
        elif not hasattr(node, "args"):
            continue
        else:
            args = node.args
        for in_node in args:
            graph["nodes"][name]["input_nodes"].append(json_creator.node_map[in_node])
            graph["nodes"][json_creator.node_map[in_node]]["output_nodes"].append(name)

    return json.dumps(graph, indent=4, sort_keys=True)


def dump_graph(mod, test_name, stage):
    if bool(int(os.environ.get("FORGE_DISABLE_REPORTIFY_DUMP", "0"))):
        return

    for global_var in mod.get_global_vars():
        mod = tvm.transform.Sequential([transform.InferType()])(mod)
        mod_fn = mod[global_var.name_hint]
        tvm_graph = convert_serialized_tvm_to_reportify_graph(mod_fn)
        path = get_default_reportify_path(test_name)
        initialize_directory(path, test_name)
        tvm_subdir = path + get_tvm_reports_relpath()
        os.makedirs(tvm_subdir + stage + "_graphs", exist_ok=True)

        filename = tvm_subdir + stage + "_" + global_var.name_hint + "_" + ".forge"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(tvm_graph)

    # assert(False)
