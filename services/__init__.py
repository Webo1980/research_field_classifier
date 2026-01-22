"""
Services Package
================
Research field classification services with multiple approaches.

Usage:
    from services import get_classifier
    
    # Get a specific approach
    classifier = get_classifier("single_shot")  # or "top_down", "embedding"
    await classifier.initialize()
    result = await classifier.classify(abstract)
    
    # Or use all approaches
    from services import get_all_classifiers
    classifiers = get_all_classifiers()
"""

from typing import Dict, Optional
from services.base.base_classifier import BaseClassifier
from services.approaches import SingleShotClassifier, TopDownClassifier, EmbeddingClassifier

# Approach registry
APPROACHES = {
    "single_shot": SingleShotClassifier,
    "top_down": TopDownClassifier,
    "embedding": EmbeddingClassifier
}


def get_classifier(approach: str = "single_shot") -> BaseClassifier:
    """
    Get a classifier instance for the specified approach.
    
    Args:
        approach: One of "single_shot", "top_down", "embedding"
        
    Returns:
        Classifier instance (not initialized)
    """
    if approach not in APPROACHES:
        raise ValueError(f"Unknown approach: {approach}. Available: {list(APPROACHES.keys())}")
    
    return APPROACHES[approach]()


def get_all_classifiers() -> Dict[str, BaseClassifier]:
    """
    Get instances of all classifier approaches.
    
    Returns:
        Dictionary mapping approach name to classifier instance
    """
    return {name: cls() for name, cls in APPROACHES.items()}


# Convenience exports
__all__ = [
    "get_classifier",
    "get_all_classifiers",
    "APPROACHES",
    "SingleShotClassifier",
    "TopDownClassifier",
    "EmbeddingClassifier"
]
