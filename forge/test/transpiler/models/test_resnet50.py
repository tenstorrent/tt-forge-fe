# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for ResNet50 model transpilation from PyTorch to ONNX to TIRGraph.
This test demonstrates:
1. Loading ResNet50 model from HuggingFace
2. Exporting to ONNX
3. Converting ONNX to TIRGraph
4. Printing graph structure with op types, shapes, dtypes, and attributes
5. Running inference and comparing outputs
"""
import pytest
import torch
import onnx
import tempfile
import os

from forge.transpiler.frontends.onnx.engine import ONNXToForgeTranspiler
from test.transpiler.test_utils import (
    print_tir_graph,
    compare_tir_with_onnx,
)


@pytest.mark.transpiler
class TestResNet50Transpilation:
    """Test cases for ResNet50 model transpilation."""

    @staticmethod
    def _load_resnet50_model():
        """Load ResNet50 model from HuggingFace."""
        try:
            from transformers import ResNetForImageClassification, AutoImageProcessor
        except ImportError:
            pytest.skip("transformers library not available")

        try:
            model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
            processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
            model.eval()
            return model, processor
        except Exception as e:
            pytest.skip(f"Failed to load ResNet50 model: {e}")

    @staticmethod
    def _create_test_input(processor=None, batch_size=1):
        """Create test input tensor for ResNet50 (224x224 RGB image)."""
        if processor is not None:
            # Use a simple test image if processor is available
            try:
                from datasets import load_dataset

                dataset = load_dataset("huggingface/cats-image", split="test")
                image = dataset[0]["image"]
                inputs = processor(image, return_tensors="pt")
                return inputs["pixel_values"]
            except Exception:
                # Fallback to random input if dataset loading fails
                pass

        # Fallback: create random RGB image (batch_size, 3, 224, 224)
        return torch.randn(batch_size, 3, 224, 224)

    @pytest.mark.nightly
    def test_resnet50_comprehensive(self):
        """
        Comprehensive test for ResNet50 model transpilation from PyTorch -> ONNX -> TIRGraph -> Forge Module.
        This test verifies the complete pipeline with both TIRGraph and Forge module generation.
        """
        print("\n" + "=" * 80)
        print("ResNet50 Model Comprehensive Transpilation Test")
        print("=" * 80)

        # Load PyTorch model
        try:
            pytorch_model, processor = self._load_resnet50_model()
        except Exception as e:
            pytest.skip(f"Failed to load ResNet50 model: {e}")

        # Create test input
        test_input = self._create_test_input(processor=processor, batch_size=1)
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)

        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
            onnx_path = tmp_file.name

        try:
            torch.onnx.export(
                pytorch_model,
                test_input,
                onnx_path,
                opset_version=17,
                input_names=["pixel_values"],
                output_names=["logits"],
                dynamic_axes=None,
                do_constant_folding=True,
            )

            # Load and validate ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"\n✓ ONNX model: opset={onnx_model.opset_import[0].version}, " f"nodes={len(onnx_model.graph.node)}")

            # # Print ONNX model structure
            # print_onnx_model(onnx_model, title="ONNX Model Structure")

            # Convert ONNX to TIRGraph (with debug mode for ONNX Runtime comparison)
            transpiler = ONNXToForgeTranspiler(validate_model=True, debug=True)
            tir_graph = transpiler.transpile(onnx_model)

            # Graph structure summary
            op_type_counts = {}
            for node in tir_graph.nodes:
                op_type_counts[node.op_type] = op_type_counts.get(node.op_type, 0) + 1

            print(
                f"\n✓ TIRGraph: {len(tir_graph.nodes)} nodes, "
                f"{len(tir_graph.params)} params, {len(tir_graph.constants)} constants"
            )
            print("Node counts by type:", ", ".join(f"{k}:{v}" for k, v in sorted(op_type_counts.items())))

            # Print detailed TIRGraph structure
            print_tir_graph(tir_graph, title="TIRGraph Structure (ResNet50 Model)", detailed=True)

            # ONNX Runtime comparison (includes shape and value verification)
            input_data = {"pixel_values": test_input.detach().cpu().numpy()}
            comparison = compare_tir_with_onnx(tir_graph, onnx_model, input_data)

            print(f"\n[ONNX Runtime Comparison]")
            if comparison["errors"]:
                print(f"  Errors ({len(comparison['errors'])}):")
                for error in comparison["errors"]:
                    print(f"    - {error}")
            else:
                print("  ✓ No errors")

            if all(comparison["matches"].values()):
                print("  ✓ All outputs match")
            else:
                print("  ⚠ Output mismatches:")
                for output_name, matches in comparison["matches"].items():
                    if not matches:
                        print(f"    - {output_name}")
                        if output_name in comparison.get("diffs", {}):
                            diff = comparison["diffs"][output_name]
                            print(f"      Max: {diff.get('max_diff', 'N/A')}, " f"Mean: {diff.get('mean_diff', 'N/A')}")

            # Part 2: Forge Module Generation
            print("\n" + "=" * 80)
            print("ResNet50 Model Forge Module Generation Test")
            print("=" * 80)

            from forge.transpiler.codegen.transpiler_to_forge import generate_forge_module_from_transpiler
            from forge.module import OnnxModule
            from forge.config import CompilerConfig
            from forge.verify.config import DeprecatedVerifyConfig

            # Create OnnxModule wrapper (reuse existing onnx_model)
            onnx_module = OnnxModule("resnet50_model", onnx_model)

            # Create compiler config with transpiler enabled
            compiler_cfg = CompilerConfig(
                compile_transpiler_to_python=True,
                transpiler_enable_debug=True,  # Enable debug mode
            )

            # Create verify config
            verify_cfg = DeprecatedVerifyConfig()
            verify_cfg.verify_forge_codegen_vs_framework = True  # Enable verification
            verify_cfg.verify_transpiler_graph = True  # Enable verification

            # Generate Forge module from transpiler
            # All verification is done inside generate_forge_module_from_transpiler
            generate_forge_module_from_transpiler(
                framework_mod=onnx_module,
                module_inputs=[test_input],
                compiler_cfg=compiler_cfg,
                graph_name="resnet50_model",
                verify_cfg=verify_cfg,
            )

            print("\n" + "=" * 80 + "\n")

        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
