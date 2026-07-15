import logging
import re
from typing import Dict, List, Set

import numpy as np
from scipy import spatial

from .tree_structures import Node

logger = logging.getLogger(__name__)


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    return [node_dict[index] for index in sorted(node_dict.keys())]


def get_embeddings(node_list: List[Node], embedding_model: str) -> List[List[float]]:
    return [node.embeddings[embedding_model] for node in node_list]


def get_text(node_list: List[Node]) -> str:
    chunks = []
    for node in node_list:
        chunks.append(" ".join(node.text.splitlines()))
    return "\n\n".join(chunks)


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
    }
    if distance_metric not in metrics:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    return [metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    return np.argsort(distances)


def split_text(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap: int = 50,
) -> List[str]:
    """Split long text into token-bounded chunks with optional overlap."""
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    token_counts = []
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        cleaned_sentences.append(sentence)
        token_counts.append(len(tokenizer.encode(sentence, add_special_tokens=False)))

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for sentence, token_count in zip(cleaned_sentences, token_counts):
        if token_count > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            words = sentence.split()
            sub_chunk: List[str] = []
            sub_length = 0
            for word in words:
                word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                if sub_length + word_tokens > max_tokens and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = sub_chunk[-max(1, overlap // 10) :]
                    sub_length = sum(
                        len(tokenizer.encode(item, add_special_tokens=False))
                        for item in sub_chunk
                    )
                sub_chunk.append(word)
                sub_length += word_tokens
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue

        if current_length + token_count > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_sentences = current_chunk[-1:] if overlap > 0 else []
            current_chunk = overlap_sentences[:]
            current_length = sum(
                len(tokenizer.encode(item, add_special_tokens=False))
                for item in current_chunk
            )

        current_chunk.append(sentence)
        current_length += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [chunk for chunk in chunks if chunk.strip()]
