from typing import Dict, List, Set


class Node:
    """A node in the RAPTOR summary tree."""

    def __init__(
        self,
        text: str,
        index: int,
        children: Set[int],
        embeddings: Dict[str, List[float]],
    ) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings


class Tree:
    """Hierarchical tree of leaf chunks and recursive summaries."""

    def __init__(
        self,
        all_nodes: Dict[int, Node],
        root_nodes: Dict[int, Node],
        leaf_nodes: Dict[int, Node],
        num_layers: int,
        layer_to_nodes: Dict[int, List[Node]],
        source_name: str = "",
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes
        self.source_name = source_name
