import torch
import torch.fx
from torch._dynamo.graph_region_tracker import GraphRegionTracker
from torch._dynamo.test_case import TestCase
from torch.utils._pytree import tree_map


def extract_graph(fn, *args, **kwargs):
    gm = None

    def extract_graph_backend(_gm, *args, **kwargs):
        nonlocal gm
        gm = _gm
        return _gm

    torch.compile(backend=extract_graph_backend)(fn)(*args, **kwargs)
    return gm.graph


def get_nodes_by_name(graph, names):
    nodes = []
    for node in graph.nodes:
        if node.name in names:
            nodes.append(node)

    return nodes


unique_ind = 0


def track_same_nodes(names, graph, region_tracker):
    global unique_ind
    unique_ind += 1
    # find nodes in graph with names and track them
    # as if they were at the same code location
    nodes = get_nodes_by_name(graph, names)
    for node in nodes:
        region_tracker.track_node("x", unique_ind, node)


class GraphRegionTrackerTests(TestCase):
    def run_test(self, fn, expected_region_groups, *args, **kwargs):
        graph = extract_graph(fn, *args, **kwargs)
        region_tracker = GraphRegionTracker()
        for region_group in expected_region_groups:
            exp_length = len(region_group[0])
            if not all(len(group) == exp_length for group in region_group):
                raise ValueError(
                    "All regions in expected region group must have the same length"
                )

            for identical_node_names in zip(*region_group):
                track_same_nodes(set(identical_node_names), graph, region_tracker)

        region_groups = region_tracker.get_identical_regions(graph)
        region_groups = tree_map(lambda n: n.name, region_groups)
        self.assertEqual(region_groups, expected_region_groups)

    def test_get_regions_single_region_group(self):
        def inner_fn(x, y):
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def fn(x, y):
            o0 = inner_fn(x, y)
            o1 = torch.sin(o0)
            o2 = inner_fn(x, o1)
            o3 = inner_fn(x, y)
            return o2 * o3 * o3

        self.run_test(
            fn,
            [
                [
                    ["y0", "x0", "sum_2", "sum_1", "z"],
                    ["y0_1", "x0_1", "sum_4", "sum_3", "z_1"],
                    ["y0_2", "x0_2", "sum_6", "sum_5", "z_2"],
                ]
            ],
            torch.rand(10, 10),
            torch.ones(10, 20),
        )

    def test_get_regions_multiple_region_groups(self):
        def inner_fn(x, y):
            x1 = x + 1
            y1 = y + 2
            z = x1.sum() + y1.sum()
            return z
        
        def inner_fn2(a, b):
            a += 2
            b += 3
            c = a * b.cos().sum()
            return c

        def fn(x, y):
            x0 = torch.cos(x)
            y0 = torch.sin(y)
            o1 = inner_fn2(x0, y0)
            o0 = inner_fn(x, y)
            o1 = torch.sin(o0)
            o2 = inner_fn(x, o1)
            o2 = inner_fn2(x0, y0)
            o3 = inner_fn(x, y)
            return o1 * o2 + o3

        self.run_test(
            fn,
            [
                [
                    ["y1", "x1", "sum_3", "sum_2", "z"],
                    ["y1_1", "x1_1", "sum_5", "sum_4", "z_1"],
                    ["y1_2", "x1_2", "sum_8", "sum_7", "z_2"],
                ],
                [
                    ["b", "cos_1", "sum_1", "a", "c"],
                    ["b_1", "cos_2", "sum_6", "a_1", "c_1"],
                ]
            ],
            torch.rand(10, 10),
            torch.ones(10, 20),
        )

    def test_no_single_node_regions(self):
        def inner_fn(x):
            return x + 1

        def fn(x):
            o0 = inner_fn(x)
            o1 = inner_fn(x)
            o2 = inner_fn(x)
            return o0 + o1 + o2

        self.run_test(fn, [], torch.ones(10, 10))

    def test_multiple_ops_on_single_line(self):
        pass

    def test_mismatched_arg_shapes(self):
        pass

    def test_mismatched_global_state(self):
        pass

    def test_no_cycles_introduced(self):
        # A cycle could be introduced if there is an external node which is a user of an internal node
        # but which also has a user in the region. Creating a single node for the region will introduce a cycle
        pass

    def test_deeply_nested_args(self):
        pass

    def test_overlapping_regions(self):
        pass

    def test_non_node_args(self):
        pass

    def test_different_input_args_for_region(self):
        pass

    def test_tensor_only_outputs(self):
        pass


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
