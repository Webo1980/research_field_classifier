"""
Classifier Approaches
=====================
Three different classification strategies for research field classification.
"""

from .single_shot import SingleShotClassifier
from .top_down import TopDownClassifier
from .embedding import EmbeddingClassifier

__all__ = [
    "SingleShotClassifier",
    "TopDownClassifier", 
    "EmbeddingClassifier"
]
