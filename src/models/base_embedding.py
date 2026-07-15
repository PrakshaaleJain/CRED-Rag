from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class BaseEmbeddingModel(ABC):
    """RAPTOR-compatible base class for text embedding models."""

    @abstractmethod
    def create_embedding(self, text: str) -> Union[List[float], np.ndarray]:
        """Return a vector representation of the input text."""
