"""
Top-Down Hierarchical Classifier (v4 - 3-Step Narrowing)
=========================================================
CRITICAL: This version uses subcategory-first narrowing.
If metadata shows "version": "v4-3step", this file is being used correctly.

Step 1: Select SUBCATEGORY from all available
Step 2: Select FIELD from ONLY that subcategory (3-18 fields, not 100+!)
"""

import asyncio
import logging
import time
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict

from config import settings
from services.base.base_classifier import BaseClassifier
from services.base.taxonomy_service import taxonomy_service
from services.base.llm_service import llm_service

logger = logging.getLogger(__name__)

# Version identifier - if you see this in metadata, the correct file is loaded
VERSION = "v4-3step"


class TopDownClassifier(BaseClassifier):
    """
    3-step hierarchical classifier with subcategory narrowing.
    
    This classifier FIRST selects a subcategory (e.g., "Computer Sciences"),
    THEN selects fields from ONLY that subcategory (typically 3-18 fields).
    
    This prevents overwhelming the LLM with 100+ fields.
    """
    
    def __init__(self):
        self.taxonomy = taxonomy_service
        self.llm = llm_service
        
        self._top_level_domains: List[str] = []
        self._all_fields: Dict[str, Dict] = {}
        
        # Hierarchy: domain -> subcategory -> [(field_id, field_name, full_line)]
        self._hierarchy: Dict[str, Dict[str, List[Tuple[str, str, str]]]] = {}
        
        # Quick lookup: subcategory -> domain
        self._subcat_to_domain: Dict[str, str] = {}
        
        self._initialized = False
        
        logger.info(f"TopDownClassifier {VERSION} instantiated")
    
    @property
    def approach_name(self) -> str:
        return "top_down"
    
    async def initialize(self):
        """Initialize taxonomy and LLM services."""
        if self._initialized:
            return
        
        logger.info(f"Initializing TopDownClassifier {VERSION}...")
        
        await self.taxonomy.initialize()
        await self.llm.initialize()
        
        # Parse taxonomy into 3-level hierarchy
        taxonomy_tree = self.taxonomy.get_compact_tree()
        self._parse_taxonomy(taxonomy_tree)
        
        self._initialized = True
        
        # Log statistics
        logger.info(f"TopDownClassifier {VERSION} initialized:")
        logger.info(f"  Domains: {len(self._top_level_domains)}")
        logger.info(f"  Total fields: {len(self._all_fields)}")
        for domain in self._top_level_domains:
            subcats = self._hierarchy.get(domain, {})
            total_fields = sum(len(fields) for fields in subcats.values())
            logger.info(f"  {domain}: {len(subcats)} subcategories, {total_fields} fields")
    
    def _parse_taxonomy(self, taxonomy_tree: str):
        """Parse taxonomy and build 3-level hierarchy."""
        top_level_set = set()
        
        for line in taxonomy_tree.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'\[([^\]]+)\]\s*(.+)', line)
            if not match:
                continue
            
            field_id = match.group(1)
            path_str = match.group(2)
            parts = [p.strip() for p in path_str.split('>')]
            
            if not parts:
                continue
            
            top_level = parts[0]
            # For 2-level paths like "Engineering > Computational Engineering",
            # the second level IS the leaf field
            second_level = parts[1] if len(parts) > 1 else top_level
            leaf_field = parts[-1]
            
            top_level_set.add(top_level)
            
            self._all_fields[field_id] = {
                'id': field_id,
                'label': leaf_field,
                'path': parts,
                'path_str': path_str,
                'top_level': top_level,
                'second_level': second_level
            }
            
            # Build 3-level hierarchy
            if top_level not in self._hierarchy:
                self._hierarchy[top_level] = {}
            if second_level not in self._hierarchy[top_level]:
                self._hierarchy[top_level][second_level] = []
            self._hierarchy[top_level][second_level].append((field_id, leaf_field, line))
            
            # Track subcategory -> domain mapping
            self._subcat_to_domain[second_level] = top_level
        
        self._top_level_domains = sorted(list(top_level_set))
    
    async def close(self):
        """Close LLM service."""
        await self.llm.close()
    
    async def classify(self, abstract: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Classify paper using 3-step approach.
        
        Step 1: Select SUBCATEGORY from all available
        Step 2: Select FIELD from that subcategory only
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.perf_counter()
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        # ==================== STEP 1: Select Subcategory ====================
        logger.info(f"[{VERSION}] Step 1: Selecting subcategory...")
        step1_start = time.perf_counter()
        
        subcategories_list = self._build_subcategories_list()
        selected_subcategory, step1_confidence, step1_tokens = await self._select_subcategory(
            abstract, subcategories_list
        )
        
        step1_time = (time.perf_counter() - step1_start) * 1000
        for k, v in step1_tokens.items():
            total_tokens[k] += v
        
        logger.info(f"[{VERSION}] Step 1 result: '{selected_subcategory}' (conf: {step1_confidence:.2f})")
        
        # ==================== STEP 2: Select Fields from Subcategory ====================
        logger.info(f"[{VERSION}] Step 2: Selecting fields from '{selected_subcategory}'...")
        step2_start = time.perf_counter()
        
        # Get ONLY fields from the selected subcategory
        subtree, field_count = self._get_fields_for_subcategory(selected_subcategory)
        
        logger.info(f"[{VERSION}] Step 2: Narrowed to {field_count} fields")
        
        annotations, step2_tokens = await self._select_fields(
            abstract, subtree, selected_subcategory, top_n
        )
        
        step2_time = (time.perf_counter() - step2_start) * 1000
        for k, v in step2_tokens.items():
            total_tokens[k] += v
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "annotations": annotations,
            "timing": {
                "total_time_ms": round(total_time, 2),
                "llm_time_ms": round(step1_time + step2_time, 2),
                "step1_subcategory_ms": round(step1_time, 2),
                "step2_field_ms": round(step2_time, 2),
                "taxonomy_lookup_ms": round(total_time - step1_time - step2_time, 2)
            },
            "token_usage": total_tokens,
            "metadata": {
                "approach": self.approach_name,
                "version": VERSION,  # <-- This identifies the file
                "selected_subcategory": selected_subcategory,
                "subcategory_confidence": step1_confidence,
                "fields_in_subcategory": field_count,
                "taxonomy_size": self.taxonomy.get_statistics()
            }
        }
    
    def _build_subcategories_list(self) -> str:
        """
        Build formatted list of ALL subcategories for Step 1 prompt.
        
        Format:
        PHYSICAL SCIENCES & MATHEMATICS:
          • Computer Sciences (18 fields): Machine Learning, Computer Vision...
          • Statistics and Probability (5 fields): Applied Statistics...
        
        LIFE SCIENCES:
          • Medicine (25 fields): Clinical Medicine, Oncology...
        """
        lines = []
        
        for domain in self._top_level_domains:
            subcats = self._hierarchy.get(domain, {})
            if not subcats:
                continue
            
            lines.append(f"\n{domain.upper()}:")
            
            # Sort by field count (most first)
            sorted_subcats = sorted(
                subcats.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            
            for subcat, fields in sorted_subcats:
                field_names = [f[1] for f in fields[:4]]  # First 4 field names
                preview = ", ".join(field_names)
                if len(fields) > 4:
                    preview += "..."
                
                lines.append(f"  • {subcat} ({len(fields)} fields): {preview}")
        
        return '\n'.join(lines)
    
    def _get_fields_for_subcategory(self, subcategory: str) -> Tuple[str, int]:
        """
        Get taxonomy lines for ONLY the fields in the selected subcategory.
        
        Returns: (subtree_text, field_count)
        """
        # Normalize subcategory name
        normalized = self._normalize_subcategory(subcategory)
        
        # Find domain
        domain = self._subcat_to_domain.get(normalized)
        
        if not domain:
            logger.warning(f"[{VERSION}] Subcategory '{subcategory}' not found, using Computer Sciences")
            normalized = "Computer Sciences"
            domain = "Physical Sciences & Mathematics"
        
        # Get fields
        fields = self._hierarchy.get(domain, {}).get(normalized, [])
        
        if not fields:
            logger.warning(f"[{VERSION}] No fields for '{normalized}', falling back to full domain")
            # Fallback: return all fields from domain
            all_lines = []
            for subcat, subfields in self._hierarchy.get(domain, {}).items():
                all_lines.extend([f[2] for f in subfields])
            return '\n'.join(all_lines[:50]), len(all_lines[:50])  # Limit to 50
        
        # Return ONLY the fields in this subcategory
        taxonomy_lines = [f[2] for f in fields]  # f[2] is the full line
        return '\n'.join(taxonomy_lines), len(taxonomy_lines)
    
    def _normalize_subcategory(self, subcategory: str) -> str:
        """Normalize subcategory name to match taxonomy."""
        if not subcategory:
            return "Computer Sciences"
        
        # Exact match
        if subcategory in self._subcat_to_domain:
            return subcategory
        
        # Fuzzy match
        subcat_lower = subcategory.lower().strip()
        for subcat in self._subcat_to_domain.keys():
            if subcat_lower == subcat.lower():
                return subcat
            if subcat_lower in subcat.lower() or subcat.lower() in subcat_lower:
                return subcat
        
        # Common mappings
        mappings = {
            "computer science": "Computer Sciences",
            "cs": "Computer Sciences",
            "ai": "Artificial Intelligence",
            "machine learning": "Artificial Intelligence",
            "medicine": "Medicine",
            "biology": "Biology/Integrated Biology/ Integrated Biomedical Sciences",
            "public health": "Public Health",
            "sociology": "Sociology",
            "economics": "Economics",
            "psychology": "Psychology",
            "physics": "Physics",
            "chemistry": "Chemistry",
            "mathematics": "Mathematics",
        }
        
        for key, value in mappings.items():
            if key in subcat_lower:
                if value in self._subcat_to_domain:
                    return value
        
        logger.warning(f"[{VERSION}] Could not normalize '{subcategory}', defaulting to Computer Sciences")
        return "Computer Sciences"
    
    async def _select_subcategory(
        self, 
        abstract: str, 
        subcategories_list: str
    ) -> Tuple[str, float, Dict]:
        """
        Step 1: Select the most relevant SUBCATEGORY.
        """
        system_prompt = """You are classifying research papers. Select the SUBCATEGORY that best matches the paper.

CRITICAL RULES:
1. Focus on the paper's MAIN CONTRIBUTION
2. If the paper develops/uses computational methods (ML, deep learning, algorithms):
   → Select "Computer Sciences" or "Artificial Intelligence" 
   NOT the application domain (Medicine, Biology, etc.)
3. Common mappings:
   - Deep learning for medical images → Computer Sciences (it's about the ML METHOD)
   - NLP/chatbot evaluation → Computer Sciences 
   - Disease epidemiology study (no new methods) → Medicine or Public Health
   - Ecosystem behavior study → Biology or related Life Sciences

Return ONLY valid JSON: {"subcategory": "Exact Name", "confidence": 0.0-1.0}"""

        user_prompt = f"""Select the best SUBCATEGORY:

AVAILABLE SUBCATEGORIES:
{subcategories_list}

PAPER:
{abstract[:2000]}

JSON response:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response, tokens = await self.llm.generate(messages, temperature=0.0, max_tokens=100)
        
        # Parse response
        selected = "Computer Sciences"
        confidence = 0.5
        
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                subcat = data.get("subcategory", "")
                confidence = float(data.get("confidence", 0.5))
                
                if subcat:
                    selected = self._normalize_subcategory(subcat)
        except Exception as e:
            logger.warning(f"[{VERSION}] Failed to parse subcategory response: {e}")
            logger.warning(f"[{VERSION}] Raw response: {response[:200]}")
        
        return selected, confidence, tokens
    
    async def _select_fields(
        self, 
        abstract: str, 
        subtree: str,
        subcategory: str,
        top_n: int
    ) -> Tuple[List[Dict], Dict]:
        """
        Step 2: Select specific FIELDS from the subcategory.
        """
        system_prompt = f"""Select the {top_n} best research fields for this paper.

You are selecting from "{subcategory}" subcategory fields ONLY.

RULES:
1. Use the EXACT field_id from the taxonomy (e.g., "R112122")
2. Choose the MOST SPECIFIC matching field
3. All selections must be from the provided taxonomy

Return ONLY valid JSON: {{"results": [{{"field_id": "R...", "field_name": "...", "score": 0.95, "reasoning": "brief"}}]}}"""

        user_prompt = f"""TAXONOMY (use exact field_id):
{subtree}

PAPER:
{abstract[:2000]}

Select {top_n} fields as JSON:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response, tokens = await self.llm.generate(messages, temperature=0.0, max_tokens=400)
        
        # Parse response
        annotations = []
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                results = data.get("results", data.get("selections", []))
                
                for sel in results[:top_n]:
                    field_id = sel.get("field_id", "")
                    if field_id in self._all_fields:
                        info = self._all_fields[field_id]
                        path_labels = self.taxonomy.get_path_labels(field_id)
                        path_ids = self.taxonomy.get_path_ids(field_id)
                        
                        if not path_labels:
                            path_labels = ["Research Field"] + info['path']
                        if not path_ids:
                            path_ids = [field_id]
                        
                        annotations.append({
                            "research_field": info['label'],
                            "research_field_id": field_id,
                            "score": float(sel.get("score", sel.get("confidence", 0.5))),
                            "reasoning": sel.get("reasoning", sel.get("reason", "")),
                            "path": path_labels,
                            "path_ids": path_ids
                        })
        except Exception as e:
            logger.warning(f"[{VERSION}] Failed to parse field response: {e}")
            logger.warning(f"[{VERSION}] Raw response: {response[:300]}")
        
        # Fallback: pad with fields from subtree if needed
        if len(annotations) < top_n:
            used_ids = {a["research_field_id"] for a in annotations}
            
            for line in subtree.strip().split('\n'):
                if len(annotations) >= top_n:
                    break
                
                match = re.match(r'\[([^\]]+)\]', line)
                if match:
                    fid = match.group(1)
                    if fid not in used_ids and fid in self._all_fields:
                        info = self._all_fields[fid]
                        path_labels = self.taxonomy.get_path_labels(fid)
                        if not path_labels:
                            path_labels = ["Research Field"] + info['path']
                        
                        annotations.append({
                            "research_field": info['label'],
                            "research_field_id": fid,
                            "score": 0.1,
                            "reasoning": "Fallback selection from subcategory",
                            "path": path_labels,
                            "path_ids": [fid]
                        })
                        used_ids.add(fid)
        
        return annotations[:top_n], tokens
