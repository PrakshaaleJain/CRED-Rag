import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure the parent directory (project root) is in the Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Ensure we can load Node and Tree classes
from src.raptor_pipeline.tree_structures import Node, Tree



def build_hierarchy(node: Node, all_nodes: Dict[int, Node]) -> Dict[str, Any]:
    """Recursively build the tree node hierarchy dictionary."""
    # Convert set of children indices to sorted list
    child_list = sorted(list(node.children))
    
    return {
        "index": node.index,
        "short_text": node.text[:80] + "..." if len(node.text) > 80 else node.text,
        "full_text": node.text,
        "children": [build_hierarchy(all_nodes[child_id], all_nodes) for child_id in child_list]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a RAPTOR pickle tree to standard JSON.")
    parser.add_argument("pkl_path", type=Path, help="Path to the tree.pkl file")
    parser.add_argument("json_path", type=Path, help="Output path for the tree.json file")
    args = parser.parse_args()

    if not args.pkl_path.exists():
        print(f"Error: Pickle file not found at {args.pkl_path}")
        return

    print(f"Loading tree from {args.pkl_path}...")
    with open(args.pkl_path, "rb") as file:
        tree = pickle.load(file)

    if not isinstance(tree, Tree):
        print("Error: Loaded object is not a RAPTOR Tree.")
        return

    print("Converting tree to hierarchical JSON structure...")
    # Map each root node to its recursive child structure
    roots = [build_hierarchy(node, tree.all_nodes) for node in tree.root_nodes.values()]

    output_data = {
        "source_name": tree.source_name,
        "num_layers": tree.num_layers,
        "num_nodes": len(tree.all_nodes),
        "roots": roots
    }

    print(f"Saving JSON to {args.json_path}...")
    with open(args.json_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    print("Success! You can now visualize this JSON in tools like JSON Crack.")


if __name__ == "__main__":
    main()
