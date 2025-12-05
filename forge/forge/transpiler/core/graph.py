"""
Graph representation for the transpiler intermediate representation.
Framework-agnostic - works for all frontends.
"""
import torch
import logging
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
from copy import deepcopy

from ..ir.nodes import TIRNode

logger = logging.getLogger("ForgeTranspiler")


class TIRGraph:
    """Represents a computational graph in Transpiler Intermediate Representation (TIR)."""
    def __init__(self, name: str, frontend_model=None, debug_mode: bool = False):
        self.name = name
        self.nodes: List[TIRNode] = []
        self.inputs: List[str] = [] 
        self.outputs: List[str] = []
        
        # Parameters vs Constants Distinction
        self.params: Dict[str, torch.Tensor] = {}      # Trainable weights (e.g., conv weights, biases)
        self.constants: Dict[str, torch.Tensor] = {}    # Non-trainable values (e.g., constant scalars, int tensors)
        
        # Topology info
        self.producer_map: Dict[str, str] = {} 
        self.consumer_map: Dict[str, List[str]] = {}
        
        # Activation memory management
        self.needed_by: Optional[Dict[str, set]] = None
        
        # Debug mode
        self.debug_mode: bool = debug_mode
        self.frontend_model = frontend_model  # Store original model for debug comparisons
        self.node_proto_map: Dict[str, Any] = {}  # Map node names to frontend node protos
    
    @property
    def initializers(self) -> Dict[str, torch.Tensor]:
        """
        Backward compatibility property that combines params and constants.
        Returns a combined dictionary of all initializers.
        """
        combined = {}
        combined.update(self.params)
        combined.update(self.constants)
        return combined
        
    def add_node(self, node: TIRNode):
        """Add a node to the graph and update topology maps."""
        self.nodes.append(node)
        # Handle multiple outputs
        for output_name in node.outputs:
            self.producer_map[output_name] = node.name
        for in_name in node.inputs:
            if in_name not in self.consumer_map:
                self.consumer_map[in_name] = []
            self.consumer_map[in_name].append(node.name)

    def get_node_by_name(self, name: str) -> Optional[TIRNode]:
        """Get a node by its name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_topological_sort(self) -> List[TIRNode]:
        """Get nodes in topological order for execution."""
        in_degree = {node.name: 0 for node in self.nodes}
        
        # Calculate in-degree based on internal graph dependencies
        for node in self.nodes:
            for input_name in node.inputs:
                # Check if input is produced by another node (i.e., not a model input/initializer)
                if input_name in self.producer_map and self.producer_map[input_name] != node.name:
                    in_degree[node.name] += 1
        
        # Initialize queue with nodes that have zero in-degree
        queue = deque([node for node in self.nodes if in_degree[node.name] == 0])
        sorted_nodes = []
        
        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            
            # Decrease in-degree of consumers
            for output_name in node.outputs:
                if output_name in self.consumer_map:
                    for consumer_name in self.consumer_map[output_name]:
                        if consumer_name in in_degree:
                            in_degree[consumer_name] -= 1
                            if in_degree[consumer_name] == 0:
                                consumer_node = self.get_node_by_name(consumer_name)
                                if consumer_node:
                                    queue.append(consumer_node)
        return sorted_nodes

    def compute_activation_dependencies(self):
        """
        Compute activation dependencies - which nodes need which activations.
        Used for memory management (garbage collection of unused activations).
        """
        needed_by = defaultdict (set)
        
        for node in self.nodes:
            out_op_id = node.outputs[0] if node.outputs else None
            if out_op_id:
                for in_op_id in node.inputs:
                    needed_by[in_op_id].add(out_op_id)
        
        needed_by.default_factory = None
        self.needed_by = dict(needed_by)
        return self.needed_by

    def run(self, inputs: Dict[str, torch.Tensor], enable_gc: bool = True) -> Dict[str, torch.Tensor]:
        """
        Execute the graph with given inputs.
        
        Args:
            inputs: Input tensors dictionary
            enable_gc: Enable activation garbage collection (memory optimization)
        """
        logger.info(f"Executing Graph: {self.name}")
        tensor_memory = {}
        # Add both params and constants to tensor memory
        tensor_memory.update(self.params)
        tensor_memory.update(self.constants)
        tensor_memory.update(inputs)
        
        # Compute dependencies if not already computed and GC is enabled
        if enable_gc and self.needed_by is None:
            self.compute_activation_dependencies()
        
        still_needed_by = deepcopy(self.needed_by) if enable_gc and self.needed_by else None
        
        # Prepare inputs for debug mode (convert to numpy)
        debug_inputs = None
        if self.debug_mode and self.frontend_model is not None:
            import numpy as np
            debug_inputs = []
            for input_name in self.inputs:
                if input_name in inputs:
                    tensor = inputs[input_name]
                    if isinstance(tensor, torch.Tensor):
                        debug_inputs.append(tensor.detach().cpu().numpy())
                    else:
                        debug_inputs.append(np.array(tensor))
        
        execution_plan = self.get_topological_sort()
        
        for node in execution_plan:
            node_inputs = {}
            ready = True
            for inp in node.inputs:
                if inp in tensor_memory:
                    node_inputs[inp] = tensor_memory[inp]
                else:
                    logger.error(f"Node {node.name} missing input: {inp}")
                    ready = False
            
            if ready:
                outputs = node.eval(node_inputs)
                tensor_memory.update(outputs)
                
                # Debug mode: compare outputs with frontend runtime (frontend-specific)
                if self.debug_mode and self.frontend_model is not None and debug_inputs is not None:
                    frontend_node = self.node_proto_map.get(node.name)
                    if frontend_node:
                        # Import debug function from frontend
                        try:
                            from ..frontends.onnx.debug.validator import debug_node_output, DebugValidationError
                            debug_node_output(
                                self.frontend_model,
                                debug_inputs,
                                outputs,
                                frontend_node
                            )
                        except DebugValidationError as e:
                            # Debug validation errors (shape/value mismatches) should stop execution
                            logger.error(
                                f"Debug validation failed for node {node.name}: {e}. "
                                f"Stopping execution in debug mode."
                            )
                            raise  # Re-raise to stop execution
                        except Exception as e:
                            # Other exceptions (e.g., ONNX Runtime errors) are logged but don't stop execution
                            logger.error(
                                f"Debug comparison failed for node {node.name}: {e}. "
                                f"This may indicate an issue with ONNX Runtime or model setup."
                            )
                
                # Garbage collection: remove activations no longer needed
                if enable_gc and still_needed_by is not None:
                    out_op_id = node.outputs[0] if node.outputs else None
                    if out_op_id:
                        for in_op_id in node.inputs:
                            if in_op_id in still_needed_by:
                                still_needed_by[in_op_id].discard(out_op_id)
                                if len(still_needed_by[in_op_id]) == 0:
                                    # Don't delete params, constants, or initializers
                                    if (in_op_id in tensor_memory and 
                                        in_op_id not in self.params and 
                                        in_op_id not in self.constants):
                                        del tensor_memory[in_op_id]
                                        logger.debug(f"GC: Deleted activation {in_op_id}")
        
        result = {}
        for out_name in self.outputs:
            if out_name in tensor_memory:
                result[out_name] = tensor_memory[out_name]
            else:
                logger.error(f"Graph output {out_name} was not produced.")
        return result

