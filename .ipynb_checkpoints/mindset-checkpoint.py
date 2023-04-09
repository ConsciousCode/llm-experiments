import ipywidgets as ipy
import networkx as nx
import torch
import matplotlib.pyplot as plt
from model import Orin

def build_model_graph(model, input_shape):
    """
    Builds a networkx graph representation of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to build a graph of.
        input_shape (tuple): The shape of the model's input tensor.

    Returns:
        nx.DiGraph: A directed graph representation of the model.
    """
    # Initialize the graph
    graph = nx.DiGraph()

    # Create a dictionary mapping tensor names to nodes in the graph
    tensor_nodes = {}

    # Add nodes for each input tensor
    for i, shape in enumerate(input_shape):
        name = f"@input_{i}"
        tensor_nodes[name] = graph.add_node(name, shape=shape)

    # Recursively add nodes for each module in the model
    def add_module(module, parent_name):
        # Create a name for the module based on its type and parent name
        name = f"{parent_name}.{type(module).__name__}"

        # Add a node for the module
        tensor_nodes[name] = graph.add_node(name)

        # Add edges from the inputs to the module
        for i, input_shape in enumerate(module.input_shapes):
            input_name = f"{name}.input_{i}"
            graph.add_edge(tensor_nodes[input_name], tensor_nodes[name])

        # Add edges from the module to the outputs
        for i, output_shape in enumerate(module.output_shapes):
            output_name = f"{name}.output_{i}"
            tensor_nodes[output_name] = graph.add_node(output_name, shape=output_shape)
            graph.add_edge(tensor_nodes[name], tensor_nodes[output_name])

        # Recursively add nodes for any child modules
        for child_name, child_module in module.named_children():
            add_module(child_module, name)

    # Add nodes for each top-level module in the model
    for name, module in model.named_children():
        add_module(module, name)

    return graph

orin = Orin.load()
G = build_model_graph(orin, orin.input_shape())
# create a graph
G = nx.Graph()
G.add_nodes_from([1, 2, 3])
G.add_edges_from([(1, 2), (2, 3)])

# draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()