import logging
from pathlib import Path
from typing import Optional, Union

from .build_tree import RaptorPipelineConfig, RaptorSummarizer, load_tree, save_tree
from .tree_structures import Tree

logger = logging.getLogger(__name__)

RetrievalAugmentationConfig = RaptorPipelineConfig


class RetrievalAugmentation:
    """
    RAPTOR-style interface for building and persisting summary trees.

    Example:
        RA = RetrievalAugmentation()
        RA.add_documents(text)
        RA.save("data/trees/HAVA_extracted")
    """

    def __init__(
        self,
        config: Optional[RaptorPipelineConfig] = None,
        tree: Optional[Union[str, Path, Tree]] = None,
    ) -> None:
        self.config = config or RaptorPipelineConfig()
        self.summarizer = RaptorSummarizer(self.config)

        if isinstance(tree, (str, Path)):
            self.tree = load_tree(tree)
        else:
            self.tree = tree

    def add_documents(self, docs: str, source_name: str = "") -> Tree:
        self.tree = self.summarizer.build_tree_from_text(docs, source_name=source_name)
        return self.tree

    def add_json_document(self, json_path: Union[str, Path]) -> Tree:
        self.tree = self.summarizer.build_tree_from_json(json_path)
        return self.tree

    def save(self, path: Union[str, Path]) -> Path:
        if self.tree is None:
            raise ValueError("No tree to save. Call add_documents() first.")
        return save_tree(self.tree, path)

    @property
    def num_layers(self) -> int:
        if self.tree is None:
            return 0
        return self.tree.num_layers

    @property
    def root_summaries(self) -> list[str]:
        if self.tree is None:
            return []
        return [node.text for node in self.tree.root_nodes.values()]
