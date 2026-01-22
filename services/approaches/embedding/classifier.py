"""
Embedding-Based Hybrid Classifier
=================================
Uses LLM for semantic understanding + embeddings for field matching.

Step 1: LLM analyzes paper and describes the research field
Step 2: Embeddings find most similar taxonomy fields

Advantages:
- Semantic matching to taxonomy
- LLM provides context and understanding
- Embeddings ensure consistency

Disadvantages:
- Requires pre-computed embeddings
- Two-step process
- Embedding similarity ≠ classification accuracy
"""

import asyncio
import logging
import time
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

from config import settings
from services.base.base_classifier import BaseClassifier
from services.base.taxonomy_service import taxonomy_service
from services.base.llm_service import llm_service
from services.base.embedding_service import embedding_service, TaxonomyEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingClassifier(BaseClassifier):
    """
    Embedding-based hybrid classifier.
    """
    
    def __init__(self):
        self.taxonomy = taxonomy_service
        self.llm = llm_service
        self.embeddings = embedding_service
        self.taxonomy_embeddings: Optional[TaxonomyEmbeddings] = None
        self._initialized = False
    
    @property
    def approach_name(self) -> str:
        return "embedding"
    
    async def initialize(self):
        if self._initialized:
            return
        
        await self.taxonomy.initialize()
        await self.llm.initialize()
        await self.embeddings.initialize()
        
        # Initialize taxonomy embeddings
        cache_dir = Path(settings.CACHE_DIR)
        tree_path = cache_dir / settings.TAXONOMY_TREE_CACHE_FILE
        
        self.taxonomy_embeddings = TaxonomyEmbeddings(self.embeddings)
        await self.taxonomy_embeddings.initialize(tree_path)
        
        self._initialized = True
        logger.info(f"EmbeddingClassifier initialized: {self.taxonomy_embeddings.stats['total_nodes']} fields")
    
    async def close(self):
        await self.llm.close()
        await self.embeddings.close()
    
    async def classify(self, abstract: str, top_n: int = 5) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        # Step 1: LLM analysis
        step1_start = time.perf_counter()
        analysis, token_usage = await self._analyze_paper(abstract)
        step1_time = (time.perf_counter() - step1_start) * 1000
        
        logger.info(f"LLM analysis: {analysis[:100]}...")
        
        # Step 2: Embedding matching
        step2_start = time.perf_counter()
        
        # Build query from analysis + abstract
        query_text = f"{analysis}\n\n{abstract[:500]}"
        query_embedding = await self.embeddings.encode_single(query_text)
        
        # Find similar fields
        similar_fields = self.taxonomy_embeddings.find_similar(query_embedding, top_n=top_n)
        
        step2_time = (time.perf_counter() - step2_start) * 1000
        
        # Build annotations
        annotations = []
        for field_id, score, info in similar_fields:
            path_labels = self.taxonomy.get_path_labels(field_id)
            path_ids = self.taxonomy.get_path_ids(field_id)
            
            if not path_labels:
                path_labels = ["Research Field"] + info.get('path', [])
            if not path_ids:
                path_ids = [field_id]
            
            annotations.append({
                "research_field": info['label'],
                "research_field_id": field_id,
                "score": score,
                "reasoning": analysis[:200],
                "path": path_labels,
                "path_ids": path_ids
            })
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "annotations": annotations,
            "timing": {
                "total_time_ms": round(total_time, 2),
                "llm_time_ms": round(step1_time, 2),
                "embedding_time_ms": round(step2_time, 2),
                "taxonomy_lookup_ms": 0
            },
            "token_usage": token_usage,
            "metadata": {
                "approach": self.approach_name,
                "llm_calls": 1,
                "llm_analysis": analysis,
                "embedding_model": settings.EMBEDDING_MODEL,
                "taxonomy_size": self.taxonomy_embeddings.stats
            }
        }
    
    async def _analyze_paper(self, abstract: str) -> tuple:
        """Use LLM to analyze paper and describe its field."""
        system_prompt = """You are a research paper classifier. Describe the research field of this paper.

Analyze:
1. The PRIMARY methodology/technique
2. The application domain
3. Main contribution type

Output 2-3 sentences describing what research field this paper belongs to.

CLASSIFICATION GUIDANCE:
- Image analysis (CT, medical imaging) → Computer Vision and Pattern Recognition
- ML algorithm development → Machine Learning
- Text processing/NLP/corpus → Computational Linguistics
- Virus research → Virology
- Nanomaterials → Nanoscience and Nanotechnology

Be specific about methodology AND application area."""

        user_prompt = f"""Describe the research field (2-3 sentences):

ABSTRACT:
{abstract[:2000]}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response, tokens = await self.llm.generate(
                messages, temperature=0.1, max_tokens=200
            )
            return response, tokens
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return abstract[:300], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
