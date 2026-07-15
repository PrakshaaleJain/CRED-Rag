import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from src.models.base_embedding import BaseEmbeddingModel
from src.models.base_summarizer import BaseSummarizationModel
from src.models.hf_embedding import HFEmbeddingModel
from src.models.qwen_summarizer import QwenSummarizationModel
from src.preprocessing.json_loader import (
    discover_extracted_json_files,
    load_concatenated_text,
)

from .tree_builder import ClusterTreeBuilder, TreeBuilderConfig
from .tree_structures import Tree

logger = logging.getLogger(__name__)


@dataclass
class RaptorPipelineConfig:
    """High-level configuration for building CRED-RAPTOR summary trees."""

    max_tokens: int = 512
    chunk_overlap: int = 50
    num_layers: int = 5
    summarization_length: int = 256
    reduction_dimension: int = 10
    cluster_threshold: float = 0.1
    summarization_model: Optional[BaseSummarizationModel] = None
    embedding_model: Optional[BaseEmbeddingModel] = None
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    summarizer_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    def to_tree_builder_config(self) -> TreeBuilderConfig:
        summarization_model = self.summarization_model or QwenSummarizationModel(
            model_name=self.summarizer_model_name
        )
        embedding_model = self.embedding_model or HFEmbeddingModel(
            model_name=self.embedding_model_name
        )

        return TreeBuilderConfig(
            max_tokens=self.max_tokens,
            chunk_overlap=self.chunk_overlap,
            num_layers=self.num_layers,
            summarization_length=self.summarization_length,
            reduction_dimension=self.reduction_dimension,
            cluster_threshold=self.cluster_threshold,
            summarization_model=summarization_model,
            embedding_model=embedding_model,
        )


class RaptorSummarizer:
    """
    Modular RAPTOR interface for building one or many summary trees.

    Each JSON file is treated as a single document by concatenating all section
    values into one string before chunking and recursive summarization.
    """

    def __init__(self, config: Optional[RaptorPipelineConfig] = None) -> None:
        self.config = config or RaptorPipelineConfig()
        self.tree_builder = ClusterTreeBuilder(self.config.to_tree_builder_config())

    def build_tree_from_text(self, text: str, source_name: str = "") -> Tree:
        return self.tree_builder.build_from_text(text, source_name=source_name)

    def build_tree_from_json(self, json_path: Union[str, Path]) -> Tree:
        json_path = Path(json_path)
        text = load_concatenated_text(json_path)
        return self.build_tree_from_text(text, source_name=json_path.stem)

    def build_trees_from_directory(
        self,
        directory: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Tree]:
        directory = Path(directory)
        trees: Dict[str, Tree] = {}

        json_files = discover_extracted_json_files(directory)
        if not json_files:
            logger.warning("No *_extracted.json files found in %s", directory)
            return trees

        for json_path in json_files:
            logger.info("Building RAPTOR tree for %s", json_path.name)
            tree = self.build_tree_from_json(json_path)
            trees[json_path.stem] = tree

            if output_dir is not None:
                save_tree(tree, Path(output_dir) / json_path.stem)

        return trees


def save_tree(tree: Tree, save_path: Union[str, Path]) -> Path:
    """Persist a built tree to disk."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    tree_file = save_path / "tree.pkl"
    with open(tree_file, "wb") as file:
        pickle.dump(tree, file)

    metadata_file = save_path / "metadata.txt"
    metadata_file.write_text(
        "\n".join(
            [
                f"source_name={tree.source_name}",
                f"num_layers={tree.num_layers}",
                f"num_nodes={len(tree.all_nodes)}",
                f"num_leaf_nodes={len(tree.leaf_nodes)}",
                f"num_root_nodes={len(tree.root_nodes)}",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("Saved tree to %s", tree_file)
    return tree_file


def load_tree(save_path: Union[str, Path]) -> Tree:
    """Load a previously saved tree."""
    save_path = Path(save_path)

    if save_path.is_dir():
        tree_file = save_path / "tree.pkl"
    else:
        tree_file = save_path

    with open(tree_file, "rb") as file:
        tree = pickle.load(file)

    if not isinstance(tree, Tree):
        raise ValueError(f"File at {tree_file} does not contain a RAPTOR Tree")

    return tree
