"""brokkr.se_float on a hand-made graph whose nodes carry the exporter's module records."""

import onnx
import pytest
from onnx import helper

from brokkr import se_float

SE = "torchvision.ops.misc.SqueezeExcitation"
TOP = ["", "features", "features.1", "features.1.block.1"]
TOP_CLASSES = ["Net", "torch.nn.modules.container.Sequential", "MBConv", SE]


def node(op, out, scopes, classes, inputs=("x",)):
    n = helper.make_node(op, list(inputs), [out])
    n.metadata_props.add(key="pkg.torch.onnx.name_scopes", value=repr(scopes + ["aten_op"]))
    n.metadata_props.add(key="pkg.torch.onnx.class_hierarchy", value=repr(classes + ["aten.op"]))
    return n


def se_graph() -> onnx.ModelProto:
    """conv (outside) -> SE: avgpool, fc1, activation, fc2, scale_activation, multiply -> conv (outside)."""
    se = "features.1.block.1"
    nodes = [
        node("Conv", "c0", TOP[:3], TOP_CLASSES[:3]),
        node("ReduceMean", "pool", TOP + [f"{se}.avgpool"], TOP_CLASSES + ["AdaptiveAvgPool2d"]),
        node("Conv", "fc1", TOP + [f"{se}.fc1"], TOP_CLASSES + ["Conv2d"]),
        node("Relu", "act", TOP + [f"{se}.activation"], TOP_CLASSES + ["ReLU"]),
        node("Conv", "fc2", TOP + [f"{se}.fc2"], TOP_CLASSES + ["Conv2d"]),
        node("Sigmoid", "scale", TOP + [f"{se}.scale_activation"], TOP_CLASSES + ["Sigmoid"]),
        node("Mul", "rescaled", TOP, TOP_CLASSES, inputs=("scale", "c0")),
        node("Conv", "c1", TOP[:3] + ["features.1.block.2"], TOP_CLASSES[:3] + ["Conv2d"]),
    ]
    graph = helper.make_graph(nodes, "g", [], [])
    return helper.make_model(graph)


def test_se_all_takes_every_node_inside_the_block():
    expected = ["pool", "fc1", "act", "fc2", "scale", "rescaled"]
    assert se_float.selected_outputs(se_graph(), "se-all") == expected


def test_se_output_takes_fc2_the_scale_activation_and_the_multiply():
    assert se_float.selected_outputs(se_graph(), "se-output") == ["fc2", "scale", "rescaled"]


def test_block_count_and_unknown_selection():
    assert se_float.se_block_count(se_graph()) == 1
    with pytest.raises(ValueError):
        se_float.in_selection(se_graph().graph.node[1], "se-everything")


def test_nodes_without_records_are_never_selected():
    bare = helper.make_node("Mul", ["a", "b"], ["m"])
    assert not se_float.in_selection(bare, "se-all") and not se_float.in_selection(bare, "se-output")
