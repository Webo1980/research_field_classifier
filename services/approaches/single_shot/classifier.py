"""
Single-Shot Classifier
======================
Classifies papers by providing the FULL taxonomy to the LLM in a single call.

Advantages:
- LLM sees ALL fields at once
- Can make globally optimal decisions
- No error propagation from hierarchical steps

Disadvantages:
- Large prompt size (~15K tokens)
- Higher cost per classification
"""

import asyncio
import logging
import time
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from config import settings
from services.base.base_classifier import BaseClassifier
from services.base.taxonomy_service import taxonomy_service
from services.base.llm_service import llm_service

logger = logging.getLogger(__name__)


class SingleShotClassifier(BaseClassifier):
    """
    Single-shot classifier that provides full taxonomy to LLM.
    """
    
    def __init__(self):
        self.taxonomy = taxonomy_service
        self.llm = llm_service
        self._taxonomy_tree: str = ""
        self._field_info: Dict[str, Dict] = {}
        self._initialized = False
    
    @property
    def approach_name(self) -> str:
        return "single_shot"
    
    async def initialize(self):
        if self._initialized:
            return
        
        await self.taxonomy.initialize()
        await self.llm.initialize()
        
        # Load taxonomy tree
        self._taxonomy_tree = self.taxonomy.get_compact_tree()
        
        # Build field info index
        for line in self._taxonomy_tree.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'\[([^\]]+)\]\s*(.+)', line)
            if match:
                field_id = match.group(1)
                path_str = match.group(2)
                parts = [p.strip() for p in path_str.split('>')]
                leaf_name = parts[-1]
                
                self._field_info[field_id] = {
                    'id': field_id,
                    'label': leaf_name,
                    'path': parts,
                    'path_str': path_str
                }
        
        self._initialized = True
        logger.info(f"SingleShotClassifier initialized: {len(self._field_info)} fields")
    
    async def close(self):
        await self.llm.close()
    
    async def classify(self, abstract: str, top_n: int = 5) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        
        system_prompt = f"""You are a research paper classifier for ORKG (Open Research Knowledge Graph).

Below is the COMPLETE taxonomy of research fields:
[FIELD_ID] Domain > Subdomain > ... > Field Name

=== TAXONOMY ===
{self._taxonomy_tree}
=== END TAXONOMY ===

Select EXACTLY {top_n} matching fields for the paper, ordered by relevance.

=== CLASSIFICATION RULES ===

RULE 1: IMAGE/VISION PAPERS
- Image analysis (CT, X-ray, photos) → [R112118] Computer Vision and Pattern Recognition
- Even if using deep learning

RULE 2: MACHINE LEARNING (only for ML METHOD papers)
- New ML algorithms/methods → [R112125] Machine Learning
- NOT for papers that just APPLY ML

RULE 3: NLP/TEXT PAPERS
- Text mining, corpus annotation, NER → [R322] Computational Linguistics

RULE 4: NANOTECHNOLOGY
- Nanomaterials, nanostructures → [R279] Nanoscience and Nanotechnology

RULE 5: VIRUS RESEARCH
- Virus mechanisms, vaccines → [R57] Virology (NOT Public Health)

RULE 6: SPECIFICITY
- Prefer specific fields over broad ones

=== OUTPUT FORMAT ===
Return EXACTLY {top_n} selections as JSON:
{{
    "selections": [
        {{"field_id": "R...", "confidence": 0.95, "reason": "brief reason"}},
        {{"field_id": "R...", "confidence": 0.85, "reason": "brief reason"}},
        {{"field_id": "R...", "confidence": 0.75, "reason": "brief reason"}},
        {{"field_id": "R...", "confidence": 0.65, "reason": "brief reason"}},
        {{"field_id": "R...", "confidence": 0.55, "reason": "brief reason"}}
    ]
}}"""

        user_prompt = f"""Classify this paper. Return EXACTLY {top_n} selections.

ABSTRACT:
{abstract[:3000]}

Return JSON with EXACTLY {top_n} field_id selections."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            llm_response, token_usage = await self.llm.generate(
                messages, temperature=0.0, max_tokens=500
            )
            llm_time = (time.perf_counter() - start_time) * 1000
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            return self._error_response(str(e))
        
        # Parse response
        annotations = self._parse_response(llm_response, top_n)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "annotations": annotations,
            "timing": {
                "total_time_ms": round(total_time, 2),
                "llm_time_ms": round(llm_time, 2),
                "taxonomy_lookup_ms": round(total_time - llm_time, 2)
            },
            "token_usage": token_usage,
            "metadata": {
                "approach": self.approach_name,
                "llm_calls": 1,
                "taxonomy_size": self.taxonomy.get_statistics()
            }
        }
    
    def _parse_response(self, response: str, top_n: int) -> List[Dict]:
        """Parse LLM response into annotations."""
        annotations = []
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                selections = data.get("selections", [])
                
                for sel in selections:
                    field_id = sel.get("field_id", "")
                    
                    if field_id in self._field_info:
                        info = self._field_info[field_id]
                        path_labels = self.taxonomy.get_path_labels(field_id)
                        path_ids = self.taxonomy.get_path_ids(field_id)
                        
                        if not path_labels:
                            path_labels = ["Research Field"] + info['path']
                        if not path_ids:
                            path_ids = [field_id]
                        
                        annotations.append({
                            "research_field": info['label'],
                            "research_field_id": field_id,
                            "score": float(sel.get("confidence", 0.5)),
                            "reasoning": sel.get("reason", ""),
                            "path": path_labels,
                            "path_ids": path_ids
                        })
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
        
        # Ensure we have top_n results
        annotations = self._pad_annotations(annotations, top_n)
        return annotations[:top_n]
    
    def _pad_annotations(self, annotations: List[Dict], top_n: int) -> List[Dict]:
        """Pad annotations to ensure top_n results."""
        if len(annotations) >= top_n:
            return annotations
        
        used_ids = {a["research_field_id"] for a in annotations}
        fallbacks = ["R11", "R12", "R279", "R277", "R197", "R112125", "R322", "R57"]
        
        for fid in fallbacks:
            if len(annotations) >= top_n:
                break
            if fid not in used_ids and fid in self._field_info:
                info = self._field_info[fid]
                annotations.append({
                    "research_field": info['label'],
                    "research_field_id": fid,
                    "score": 0.1,
                    "reasoning": "Fallback",
                    "path": ["Research Field"] + info['path'],
                    "path_ids": [fid]
                })
                used_ids.add(fid)
        
        return annotations
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            "annotations": [],
            "timing": {"total_time_ms": 0, "llm_time_ms": 0, "taxonomy_lookup_ms": 0},
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "metadata": {"approach": self.approach_name, "error": error}
        }
