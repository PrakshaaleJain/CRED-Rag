from pathlib import Path
from typing import Union

from .build_tree import save_tree as _save_tree
from .tree_structures import Tree


def save_tree(tree: Tree, save_path: Union[str, Path]) -> Path:
    return _save_tree(tree, save_path)
