import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.models.base_embedding import BaseEmbeddingModel
from src.models.base_summarizer import BaseSummarizationModel
from src.models.hf_embedding import HFEmbeddingModel
from src.models.qwen_summarizer import QwenSummarizationModel

from .cluster_utils import RAPTORClustering
from .tree_structures import Node, Tree
from .utils import get_node_list, get_text, split_text

logger = logging.getLogger(__name__)


@dataclass
class TreeBuilderConfig:
    max_tokens: int = 512
    chunk_overlap: int = 50
    num_layers: int = 5
    summarization_length: int = 256
    reduction_dimension: int = 10
    cluster_threshold: float = 0.1
    max_length_in_cluster: int = 3500
    summarization_model: BaseSummarizationModel = field(default_factory=QwenSummarizationModel)
    embedding_model: BaseEmbeddingModel = field(default_factory=HFEmbeddingModel)
    cluster_embedding_model: str = "BGE"
    tokenizer = None

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = self.summarization_model.tokenizer

        if not isinstance(self.summarization_model, BaseSummarizationModel):
            raise ValueError("summarization_model must extend BaseSummarizationModel")

        if not isinstance(self.embedding_model, BaseEmbeddingModel):
            raise ValueError("embedding_model must extend BaseEmbeddingModel")


class ClusterTreeBuilder:
    """Build a recursive summary tree using RAPTOR-style GMM clustering."""

    def __init__(self, config: Optional[TreeBuilderConfig] = None) -> None:
        self.config = config or TreeBuilderConfig()
        self.tokenizer = self.config.tokenizer
        self.max_tokens = self.config.max_tokens
        self.num_layers = self.config.num_layers
        self.summarization_length = self.config.summarization_length
        self.summarization_model = self.config.summarization_model
        self.embedding_model = self.config.embedding_model
        self.cluster_embedding_model = self.config.cluster_embedding_model

    def summarize(self, context: str, max_tokens: Optional[int] = None) -> str:
        return self.summarization_model.summarize(
            context,
            max_tokens=max_tokens or self.summarization_length,
        )

    def create_node(
        self,
        index: int,
        text: str,
        children_indices: Optional[Set[int]] = None,
    ) -> tuple[int, Node]:
        if children_indices is None:
            children_indices = set()

        embeddings = {
            self.cluster_embedding_model: self.embedding_model.create_embedding(text)
        }
        return index, Node(text, index, children_indices, embeddings)

    def _create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        leaf_nodes: Dict[int, Node] = {}
        for index, chunk in enumerate(chunks):
            _, node = self.create_node(index, chunk)
            leaf_nodes[index] = node
        return leaf_nodes

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
    ) -> Dict[int, Node]:
        next_node_index = len(all_tree_nodes)

        for layer in range(self.num_layers):
            logger.info("Constructing layer %s with %s nodes", layer, len(current_level_nodes))

            node_list = get_node_list(current_level_nodes)
            if len(node_list) <= self.config.reduction_dimension + 1:
                self.num_layers = layer
                logger.info("Stopping at layer %s; not enough nodes to cluster further", layer)
                break

            clusters = RAPTORClustering.perform_clustering(
                node_list,
                self.cluster_embedding_model,
                max_length_in_cluster=self.config.max_length_in_cluster,
                tokenizer=self.tokenizer,
                reduction_dimension=self.config.reduction_dimension,
                threshold=self.config.cluster_threshold,
            )

            new_level_nodes: Dict[int, Node] = {}
            for cluster in clusters:
                node_texts = get_text(cluster)
                summarized_text = self.summarize(
                    context=node_texts,
                    max_tokens=self.summarization_length,
                )
                _, parent_node = self.create_node(
                    next_node_index,
                    summarized_text,
                    {node.index for node in cluster},
                )
                new_level_nodes[next_node_index] = parent_node
                next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            if len(new_level_nodes) <= 1:
                self.num_layers = layer + 1
                logger.info("Converged to a single root node at layer %s", layer + 1)
                break

        return current_level_nodes

    def build_from_text(self, text: str, source_name: str = "") -> Tree:
        chunks = split_text(
            text,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            overlap=self.config.chunk_overlap,
        )

        if not chunks:
            raise ValueError("No text chunks were produced from the input document")

        logger.info("Created %s leaf chunks", len(chunks))

        leaf_nodes = self._create_leaf_nodes(chunks)
        layer_to_nodes = {0: list(leaf_nodes.values())}
        all_nodes = copy.deepcopy(leaf_nodes)
        root_nodes = self.construct_tree(leaf_nodes, all_nodes, layer_to_nodes)

        return Tree(
            all_nodes=all_nodes,
            root_nodes=root_nodes,
            leaf_nodes=leaf_nodes,
            num_layers=self.num_layers,
            layer_to_nodes=layer_to_nodes,
            source_name=source_name,
        )
