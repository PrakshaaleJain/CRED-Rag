#!/usr/bin/env python3
"""Build RAPTOR summary trees for one or more extracted SEC JSON files."""

import argparse
import logging
from pathlib import Path

from src.raptor_pipeline import RaptorPipelineConfig, RaptorSummarizer


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_input = project_root / "data" / "sec_extracted_text"
    default_output = project_root / "data" / "trees"

    parser = argparse.ArgumentParser(
        description="Build RAPTOR summary trees from extracted SEC JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing *_extracted.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where built trees are saved",
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=None,
        help="Build a tree for a single JSON file instead of the whole directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per leaf chunk",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=5,
        help="Maximum number of recursive tree layers",
    )
    parser.add_argument(
        "--summarizer-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model id for summarization",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace model id for embeddings",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    config = RaptorPipelineConfig(
        max_tokens=args.max_tokens,
        num_layers=args.num_layers,
        summarizer_model_name=args.summarizer_model,
        embedding_model_name=args.embedding_model,
    )
    summarizer = RaptorSummarizer(config)

    if args.json_file is not None:
        tree = summarizer.build_tree_from_json(args.json_file)
        from src.raptor_pipeline.build_tree import save_tree

        save_tree(tree, args.output_dir / args.json_file.stem)
        return

    summarizer.build_trees_from_directory(args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
