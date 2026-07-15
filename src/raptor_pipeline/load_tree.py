from pathlib import Path
from typing import Union

from .build_tree import load_tree as _load_tree
from .tree_structures import Tree


def load_tree(save_path: Union[str, Path]) -> Tree:
    return _load_tree(save_path)
