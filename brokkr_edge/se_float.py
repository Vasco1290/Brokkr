"""Which nodes of a model belong to its squeeze-and-excitation blocks (the SE1 diagnostic)?

A squeeze-and-excitation block (torchvision.ops.misc.SqueezeExcitation) is a small side branch that
turns each channel up or down: average-pool -> 1x1 convolution (fc1) -> activation -> 1x1 convolution
(fc2) -> scale activation (Sigmoid or Hardsigmoid) -> multiply the block's input by the result.

The nodes are found from what the exporter recorded on each node of the FP32 ONNX file (the module path
"pkg.torch.onnx.name_scopes" and the module classes "pkg.torch.onnx.class_hierarchy"), never from tensor
names. Two selections (docs/hypotheses_stage4.md, SE1):
    se-all:    every node inside a squeeze-and-excitation module;
    se-output: only its output path: fc2, the scale activation, and the multiply (the only operation
               in the block's own forward, outside its submodules).
The result is a list of output tensor names, which survive ONNX Runtime's preparation step, so the
nodes can be found again in the prepared model that is quantized.
"""

import ast

import onnx

SE_CLASS = "torchvision.ops.misc.SqueezeExcitation"
SELECTIONS = ("se-all", "se-output")


def module_path(node: onnx.NodeProto) -> tuple:
    """(module paths, module classes) from the outermost module in; the final aten op is dropped."""
    md = {p.key: p.value for p in node.metadata_props}
    scopes = ast.literal_eval(md.get("pkg.torch.onnx.name_scopes", "[]"))
    classes = ast.literal_eval(md.get("pkg.torch.onnx.class_hierarchy", "[]"))
    return scopes[:-1], classes[:-1]


def in_selection(node: onnx.NodeProto, selection: str) -> bool:
    scopes, classes = module_path(node)
    if SE_CLASS not in classes:
        return False
    if selection == "se-all":
        return True
    if selection != "se-output":
        raise ValueError(f"unknown selection {selection!r}; expected one of {SELECTIONS}")
    j = len(classes) - 1 - classes[::-1].index(SE_CLASS)  # the innermost squeeze-and-excitation module
    se = scopes[j]
    if j == len(scopes) - 1:  # in the block's own forward: the multiply
        return node.op_type == "Mul"
    return scopes[j + 1] in (f"{se}.fc2", f"{se}.scale_activation")


def selected_outputs(model: onnx.ModelProto, selection: str) -> list:
    """Output tensor names of the selected nodes, in graph order."""
    return [out for node in model.graph.node if in_selection(node, selection) for out in node.output]


def se_block_count(model: onnx.ModelProto) -> int:
    """How many squeeze-and-excitation modules the export records (for the build record)."""
    blocks = set()
    for node in model.graph.node:
        scopes, classes = module_path(node)
        blocks.update(s for s, c in zip(scopes, classes, strict=True) if c == SE_CLASS)
    return len(blocks)
