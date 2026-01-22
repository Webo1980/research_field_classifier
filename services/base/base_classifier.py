"""
Base Classifier Interface
=========================
Abstract base class that all classifier approaches must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseClassifier(ABC):
    """
    Abstract base class for research field classifiers.
    
    All classifier approaches (single_shot, top_down, embedding) must implement this interface.
    """
    
    @property
    @abstractmethod
    def approach_name(self) -> str:
        """Return the name of this approach."""
        pass
    
    @abstractmethod
    async def initialize(self):
        """Initialize the classifier and load required resources."""
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass
    
    @abstractmethod
    async def classify(self, abstract: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Classify a research paper abstract.
        
        Args:
            abstract: The paper abstract text
            top_n: Number of top predictions to return
            
        Returns:
            Dictionary with:
                - annotations: List of classification results
                - timing: Timing information
                - token_usage: Token usage statistics
                - metadata: Additional metadata
        """
        pass
    
    async def classify_batch(
        self,
        abstracts: List[str],
        top_n: int = 5,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple papers.
        
        Default implementation processes sequentially.
        Subclasses can override for parallel processing.
        """
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def classify_one(abstract: str):
            async with semaphore:
                return await self.classify(abstract, top_n)
        
        tasks = [classify_one(a) for a in abstracts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            r if not isinstance(r, Exception) else {"error": str(r), "annotations": []}
            for r in results
        ]
