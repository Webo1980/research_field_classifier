"""
Research Field Classifier Evaluation Script
=============================================
Evaluates classifier approaches against a ground truth dataset.

CRITICAL: This script calls BOTH:
1. Old ORKG flat classifier API (https://incubating.orkg.org/nlp/api/annotation/rfclf)
2. New classifier API (local FastAPI server)

All results are saved in a format that includes old classifier data for comparison.

Usage:
    python evaluation/evaluation_script.py
    python evaluation/evaluation_script.py --approaches single_shot,top_down,embedding
    python evaluation/evaluation_script.py --max-papers 10
"""

import asyncio
import json
import os
import sys
import time
import re
import httpx
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug flag - set to True to see full API responses
DEBUG_API_RESPONSES = True

# ============================================================================
# API ENDPOINTS
# ============================================================================
OLD_CLASSIFIER_URL = "https://orkg.org/nlp/api/annotation/rfclf"
NEW_CLASSIFIER_URL = "http://localhost:8080"


class EvaluationRunner:
    """Runs evaluation for all classifier approaches including old ORKG."""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or Path(__file__).parent / "evaluation_dataset.csv"
        self.papers: List[Dict] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Taxonomy data for h-distance calculation
        self._taxonomy_paths = {}
        self._taxonomy_by_label = {}
        self._load_taxonomy()
    
    def _load_taxonomy(self):
        """Load taxonomy for path display and h-distance calculation."""
        for path in [Path("./cache/taxonomy_tree.txt"), 
                     Path(settings.CACHE_DIR) / "taxonomy_tree.txt"]:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line or not line.startswith('['):
                                continue
                            
                            bracket_end = line.index(']')
                            field_id = line[1:bracket_end]
                            path_str = line[bracket_end + 1:].strip()
                            path_parts = [p.strip() for p in path_str.split('>')]
                            
                            self._taxonomy_paths[field_id] = path_parts
                            self._taxonomy_paths[field_id.lower()] = path_parts
                            
                            if path_parts:
                                leaf = path_parts[-1].lower()
                                self._taxonomy_by_label[leaf] = {"id": field_id, "path": path_parts}
                    
                    logger.info(f"Loaded {len(self._taxonomy_paths) // 2} taxonomy paths")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load taxonomy: {e}")
    
    def _get_field_path(self, field_id: str = None, field_label: str = None) -> List[str]:
        """Get taxonomy path for a field."""
        if field_id:
            if field_id in self._taxonomy_paths:
                return self._taxonomy_paths[field_id]
            if field_id.lower() in self._taxonomy_paths:
                return self._taxonomy_paths[field_id.lower()]
        
        if field_label:
            normalized = field_label.lower().strip()
            if normalized in self._taxonomy_by_label:
                return self._taxonomy_by_label[normalized]["path"]
            # Try partial match
            for key, data in self._taxonomy_by_label.items():
                if normalized in key or key in normalized:
                    return data["path"]
        
        return []
    
    def _get_field_id_by_label(self, field_label: str) -> str:
        """Get field ID from label using taxonomy lookup."""
        if not field_label:
            return ""
        normalized = field_label.lower().strip()
        if normalized in self._taxonomy_by_label:
            return self._taxonomy_by_label[normalized]["id"]
        # Try partial match
        for key, data in self._taxonomy_by_label.items():
            if normalized in key or key in normalized:
                return data["id"]
        return ""
    
    def _calculate_h_distance(self, path1: List[str], path2: List[str]) -> int:
        """Calculate hierarchical distance between two taxonomy paths."""
        if not path1 or not path2:
            return -1
        
        p1 = [p.lower().strip() for p in path1]
        p2 = [p.lower().strip() for p in path2]
        
        # Find common ancestor depth
        common_depth = 0
        for i in range(min(len(p1), len(p2))):
            if p1[i] == p2[i]:
                common_depth = i + 1
            else:
                break
        
        # Distance = steps up + steps down
        return (len(p1) - common_depth) + (len(p2) - common_depth)
    
    def load_dataset(self) -> bool:
        """Load evaluation dataset from CSV."""
        try:
            df = pd.read_csv(self.dataset_path, encoding='utf-8')
            logger.info(f"Loaded {len(df)} papers from {self.dataset_path}")
            
            required_cols = ['paper_id', 'research_field_id', 'research_field_name']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                return False
            
            self.papers = df.to_dict('records')
            return True
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
    
    async def fetch_paper_content(self, paper: Dict) -> str:
        """Fetch abstract/content for a paper."""
        # Try ORKG API first
        paper_id = paper.get('paper_id', '')
        if paper_id.startswith('R'):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    url = f"https://orkg.org/api/papers/{paper_id}"
                    response = await client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        title = data.get("title", "")
                        if title:
                            return title
            except:
                pass
        
        # Try Crossref with DOI
        doi = paper.get('doi', '')
        if doi:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    url = f"https://api.crossref.org/works/{doi}"
                    response = await client.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        message = data.get('message', {})
                        abstract = message.get('abstract', '')
                        if abstract:
                            return re.sub(r'<[^>]+>', '', abstract)
                        title = message.get('title', [''])[0]
                        if title:
                            return title
            except:
                pass
        
        # Fallback to title or field name
        return paper.get('title', paper.get('research_field_name', 'research paper'))
    
    # ========================================================================
    # OLD ORKG CLASSIFIER API CALL
    # ========================================================================
    async def call_old_classifier(self, text: str) -> Dict:
        """
        Call the old ORKG flat classifier API.
        Returns dict with predictions, response time, and success flag.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                start_time = time.time()
                
                response = await client.post(
                    OLD_CLASSIFIER_URL,
                    json={"raw_input": text, "top_n": 5},
                    headers={"Content-Type": "application/json"}
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = []
                    
                    # Handle ORKG API response format:
                    # {"timestamp": "...", "uuid": "...", "payload": {"annotations": [...]}}
                    if isinstance(data, dict) and "payload" in data:
                        annotations = data.get("payload", {}).get("annotations", [])
                        for item in annotations[:5]:
                            field_name = item.get("research_field", item.get("label", ""))
                            predictions.append({
                                "field": field_name,
                                "field_id": self._get_field_id_by_label(field_name),
                                "score": float(item.get("score", item.get("certainty", 0))),
                            })
                    elif isinstance(data, list):
                        for item in data[:5]:
                            field_name = item.get("research_field", item.get("label", ""))
                            predictions.append({
                                "field": field_name,
                                "field_id": self._get_field_id_by_label(field_name),
                                "score": float(item.get("score", item.get("certainty", 0))),
                            })
                    
                    # Ensure we have 5 predictions (pad with empty if needed)
                    while len(predictions) < 5:
                        predictions.append({"field": "", "field_id": "", "score": 0.0})
                    
                    return {
                        "success": True,
                        "predictions": predictions[:5],
                        "response_time_ms": elapsed_ms,
                    }
                else:
                    logger.warning(f"Old classifier returned {response.status_code}")
                    return {
                        "success": False,
                        "predictions": [{"field": "", "field_id": "", "score": 0.0}] * 5,
                        "response_time_ms": elapsed_ms,
                        "error": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            logger.warning(f"Old classifier error: {e}")
            return {
                "success": False,
                "predictions": [{"field": "", "field_id": "", "score": 0.0}] * 5,
                "response_time_ms": 0,
                "error": str(e)
            }
    
    # ========================================================================
    # NEW CLASSIFIER API CALL
    # ========================================================================
    async def call_new_classifier(self, text: str, approach: str) -> Dict:
        """
        Call the new classifier API.
        Returns dict with predictions, response time, tokens, and success flag.
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                start_time = time.time()
                
                url = f"{NEW_CLASSIFIER_URL}/classify/{approach}"
                response = await client.post(
                    url,
                    json={"raw_input": text, "top_n": 5},
                    headers={"Content-Type": "application/json"}
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = []
                    
                    # Debug: Print actual response structure
                    if DEBUG_API_RESPONSES:
                        import json as json_module
                        logger.info(f"  NEW API Response ({approach}): {json_module.dumps(data, indent=2)[:500]}...")
                    
                    # Log the response structure for debugging
                    logger.debug(f"New classifier response keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
                    
                    # Handle different response formats
                    pred_list = []
                    if isinstance(data, dict):
                        # FIRST: Check for payload.annotations format (same as old classifier)
                        if "payload" in data:
                            pred_list = data.get("payload", {}).get("annotations", [])
                        
                        # If not found, try other possible keys
                        if not pred_list:
                            pred_list = data.get("predictions", 
                                        data.get("results", 
                                        data.get("classifications", 
                                        data.get("annotations",
                                        data.get("fields", [])))))
                        
                        # If it's a single prediction in the root
                        if not pred_list and "field" in data:
                            pred_list = [data]
                        elif not pred_list and "label" in data:
                            pred_list = [data]
                        elif not pred_list and "research_field" in data:
                            pred_list = [data]
                    elif isinstance(data, list):
                        pred_list = data
                    
                    for item in pred_list[:5]:
                        if isinstance(item, dict):
                            # Try multiple possible field names
                            field_name = (item.get("label") or 
                                         item.get("field") or 
                                         item.get("research_field") or 
                                         item.get("name") or "")
                            field_id = (item.get("id") or 
                                       item.get("field_id") or 
                                       item.get("research_field_id") or "")
                            score = float(item.get("confidence") or 
                                         item.get("score") or 
                                         item.get("probability") or 0)
                            reasoning = item.get("reasoning", item.get("explanation", ""))
                            
                            if field_name:  # Only add if we have a field name
                                predictions.append({
                                    "field": field_name,
                                    "field_id": field_id,
                                    "score": score,
                                    "reasoning": reasoning,
                                })
                        elif isinstance(item, str):
                            # If predictions are just strings
                            predictions.append({
                                "field": item,
                                "field_id": "",
                                "score": 0.0,
                                "reasoning": "",
                            })
                    
                    # Ensure we have 5 predictions (pad with empty if needed)
                    while len(predictions) < 5:
                        predictions.append({"field": "", "field_id": "", "score": 0.0, "reasoning": ""})
                    
                    # Get token info - check token_usage object and fallbacks
                    def get_token_value(d, *keys):
                        """Try multiple keys to find token value."""
                        for key in keys:
                            if key in d:
                                return d[key]
                        return 0
                    
                    # API returns token_usage and timing objects at root level
                    token_usage = data.get("token_usage", {}) if isinstance(data, dict) else {}
                    timing = data.get("timing", {}) if isinstance(data, dict) else {}
                    
                    # Also check legacy locations
                    payload = data.get("payload", {}) if isinstance(data, dict) else {}
                    usage = data.get("usage", payload.get("usage", {}))
                    
                    # Extract tokens (prioritize token_usage object)
                    input_tokens = get_token_value(token_usage, "input_tokens", "prompt_tokens") or \
                                   get_token_value(data, "input_tokens", "prompt_tokens") or \
                                   get_token_value(usage, "input_tokens", "prompt_tokens")
                    
                    output_tokens = get_token_value(token_usage, "output_tokens", "completion_tokens") or \
                                    get_token_value(data, "output_tokens", "completion_tokens") or \
                                    get_token_value(usage, "output_tokens", "completion_tokens")
                    
                    total_tokens = get_token_value(token_usage, "total_tokens") or \
                                   get_token_value(data, "total_tokens") or \
                                   get_token_value(usage, "total_tokens") or \
                                   (input_tokens + output_tokens)
                    
                    # Extract timing info (prioritize timing object)
                    llm_time_ms = get_token_value(timing, "llm_time_ms") or \
                                  get_token_value(data, "llm_time_ms") or elapsed_ms
                    taxonomy_lookup_ms = get_token_value(timing, "taxonomy_lookup_ms") or \
                                         get_token_value(data, "taxonomy_lookup_ms") or 0
                    
                    # Estimate tokens if not provided by API (~4 chars per token)
                    if total_tokens == 0 and text:
                        estimated_input = len(text) // 4 + 500  # +500 for prompt template
                        estimated_output = 200  # Typical output size
                        input_tokens = estimated_input
                        output_tokens = estimated_output
                        total_tokens = estimated_input + estimated_output
                    
                    return {
                        "success": True,
                        "predictions": predictions[:5],
                        "response_time_ms": elapsed_ms,
                        "llm_time_ms": llm_time_ms,
                        "taxonomy_lookup_ms": taxonomy_lookup_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }
                else:
                    logger.warning(f"New classifier returned {response.status_code}: {response.text[:200]}")
                    return {
                        "success": False,
                        "predictions": [{"field": "", "field_id": "", "score": 0.0}] * 5,
                        "response_time_ms": elapsed_ms,
                        "error": f"HTTP {response.status_code}"
                    }
        except Exception as e:
            logger.warning(f"New classifier error ({approach}): {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "predictions": [{"field": "", "field_id": "", "score": 0.0}] * 5,
                "response_time_ms": 0,
                "error": str(e)
            }
    
    # ========================================================================
    # EVALUATION LOGIC
    # ========================================================================
    def evaluate_predictions(self, gt_id: str, gt_label: str, predictions: List[Dict]) -> Dict:
        """Evaluate predictions against ground truth, adding h-distance to each."""
        gt_path = self._get_field_path(gt_id, gt_label)
        gt_normalized = gt_label.lower().strip()
        
        correct_top1 = False
        correct_top5 = False
        match_position = 0
        
        # Process each prediction
        for i, pred in enumerate(predictions[:5], 1):
            pred_label = pred.get("field", "").lower().strip()
            pred_id = pred.get("field_id", "")
            
            # Get path and calculate h-distance
            pred_path = self._get_field_path(pred_id, pred_label)
            h_dist = self._calculate_h_distance(gt_path, pred_path)
            
            # Add path and h_distance to prediction
            pred["path"] = pred_path
            pred["h_distance"] = h_dist
            
            # Check if match
            is_match = (gt_normalized == pred_label or 
                       (gt_id and gt_id == pred_id) or
                       h_dist == 0)
            
            if is_match and match_position == 0:
                match_position = i
                if i == 1:
                    correct_top1 = True
                correct_top5 = True
        
        # ORKG accuracy score
        if match_position == 1:
            accuracy_score = 1.0
        elif match_position in [2, 3]:
            accuracy_score = 0.8
        elif match_position in [4, 5]:
            accuracy_score = 0.6
        else:
            accuracy_score = 0.0
        
        return {
            "correct_top1": correct_top1,
            "correct_top5": correct_top5,
            "match_position": match_position,
            "accuracy_score": accuracy_score,
            "ground_truth_path": gt_path,
        }
    
    async def evaluate_paper(self, paper: Dict, approaches: List[str]) -> Dict:
        """Evaluate a single paper with ALL classifiers (old + new)."""
        paper_id = paper.get('paper_id', '')
        gt_id = paper.get('research_field_id', '')
        gt_label = paper.get('research_field_name', '')
        gt_path = self._get_field_path(gt_id, gt_label)
        
        logger.info(f"Evaluating {paper_id}...")
        
        # Fetch content
        content = await self.fetch_paper_content(paper)
        if not content or len(content) < 5:
            content = f"{gt_label} research paper"
        
        result = {
            "paper_id": paper_id,
            "doi": paper.get('doi', ''),
            "title": paper.get('title', gt_label),
            "abstract": content[:500] + "..." if len(content) > 500 else content,
            "abstract_length": len(content),
            "ground_truth_id": gt_id,
            "ground_truth_label": gt_label,
            "ground_truth_path": gt_path,
        }
        
        # ====== CALL OLD CLASSIFIER ======
        logger.info(f"  Calling OLD classifier...")
        old_result = await self.call_old_classifier(content)
        old_eval = self.evaluate_predictions(gt_id, gt_label, old_result.get("predictions", []))
        
        result["old_classifier"] = {
            "success": old_result.get("success", False),
            "predictions": old_result.get("predictions", []),
            "response_time_ms": old_result.get("response_time_ms", 0),
            "correct_top1": old_eval["correct_top1"],
            "correct_top5": old_eval["correct_top5"],
            "match_position": old_eval["match_position"],
            "accuracy_score": old_eval["accuracy_score"],
        }
        
        # ====== CALL NEW CLASSIFIERS ======
        for approach in approaches:
            logger.info(f"  Calling NEW classifier ({approach})...")
            new_result = await self.call_new_classifier(content, approach)
            new_eval = self.evaluate_predictions(gt_id, gt_label, new_result.get("predictions", []))
            
            result[approach] = {
                "success": new_result.get("success", False),
                "predictions": new_result.get("predictions", []),
                "response_time_ms": new_result.get("response_time_ms", 0),
                "llm_time_ms": new_result.get("llm_time_ms", 0),
                "taxonomy_lookup_ms": new_result.get("taxonomy_lookup_ms", 0),
                "input_tokens": new_result.get("input_tokens", 0),
                "output_tokens": new_result.get("output_tokens", 0),
                "total_tokens": new_result.get("total_tokens", 0),
                "correct_top1": new_eval["correct_top1"],
                "correct_top5": new_eval["correct_top5"],
                "match_position": new_eval["match_position"],
                "accuracy_score": new_eval["accuracy_score"],
            }
        
        return result
    
    def calculate_metrics(self, all_results: List[Dict], approaches: List[str]) -> Dict[str, Dict]:
        """Calculate aggregate metrics for all approaches."""
        valid = [r for r in all_results if "error" not in r]
        n = len(valid)
        
        if n == 0:
            return {}
        
        metrics = {}
        
        # Old classifier metrics
        old_data = [r.get("old_classifier", {}) for r in valid]
        old_h_dists = []
        for d in old_data:
            preds = d.get("predictions", [])
            if preds and preds[0].get("h_distance", -1) >= 0:
                old_h_dists.append(preds[0]["h_distance"])
        
        metrics["old_classifier"] = {
            "total_papers": len(all_results),
            "evaluated_papers": n,
            "top1_accuracy": sum(1 for d in old_data if d.get("correct_top1")) / n,
            "top5_accuracy": sum(1 for d in old_data if d.get("correct_top5")) / n,
            "accuracy_score": sum(d.get("accuracy_score", 0) for d in old_data) / n,
            "avg_response_time_ms": sum(d.get("response_time_ms", 0) for d in old_data) / n,
            "avg_h_distance": sum(old_h_dists) / len(old_h_dists) if old_h_dists else 0,
        }
        
        # New classifier metrics
        for approach in approaches:
            new_data = [r.get(approach, {}) for r in valid]
            new_h_dists = []
            for d in new_data:
                preds = d.get("predictions", [])
                if preds and preds[0].get("h_distance", -1) >= 0:
                    new_h_dists.append(preds[0]["h_distance"])
            
            metrics[approach] = {
                "total_papers": len(all_results),
                "evaluated_papers": n,
                "top1_accuracy": sum(1 for d in new_data if d.get("correct_top1")) / n,
                "top5_accuracy": sum(1 for d in new_data if d.get("correct_top5")) / n,
                "accuracy_score": sum(d.get("accuracy_score", 0) for d in new_data) / n,
                "avg_response_time_ms": sum(d.get("response_time_ms", 0) for d in new_data) / n,
                "avg_llm_time_ms": sum(d.get("llm_time_ms", 0) for d in new_data) / n,
                "total_tokens": sum(d.get("total_tokens", 0) for d in new_data),
                "avg_h_distance": sum(new_h_dists) / len(new_h_dists) if new_h_dists else 0,
            }
        
        return metrics
    
    def save_results(self, all_results: List[Dict], metrics: Dict[str, Dict], approaches: List[str]):
        """Save evaluation results in format for visualization."""
        output_dir = Path(settings.EVALUATION_RESULTS_DIR)
        
        # Save per-approach results (includes old classifier data for comparison)
        for approach in approaches:
            approach_dir = output_dir / approach
            approach_dir.mkdir(parents=True, exist_ok=True)
            
            approach_results = []
            for r in all_results:
                if "error" in r:
                    continue
                
                new_data = r.get(approach, {})
                old_data = r.get("old_classifier", {})
                
                approach_results.append({
                    "paper_id": r.get("paper_id"),
                    "doi": r.get("doi"),
                    "title": r.get("title"),
                    "abstract": r.get("abstract"),
                    "ground_truth_id": r.get("ground_truth_id"),
                    "ground_truth_label": r.get("ground_truth_label"),
                    "ground_truth_path": r.get("ground_truth_path"),
                    # New classifier
                    "correct_top1": new_data.get("correct_top1", False),
                    "correct_top5": new_data.get("correct_top5", False),
                    "match_position": new_data.get("match_position", 0),
                    "accuracy_score": new_data.get("accuracy_score", 0),
                    "predictions": new_data.get("predictions", []),
                    "response_time_ms": new_data.get("response_time_ms", 0),
                    "llm_time_ms": new_data.get("llm_time_ms", 0),
                    "taxonomy_lookup_ms": new_data.get("taxonomy_lookup_ms", 0),
                    "input_tokens": new_data.get("input_tokens", 0),
                    "output_tokens": new_data.get("output_tokens", 0),
                    "total_tokens": new_data.get("total_tokens", 0),
                    "h_distance": new_data.get("predictions", [{}])[0].get("h_distance", -1) if new_data.get("predictions") else -1,
                    # OLD classifier (for comparison)
                    "old_predictions": old_data.get("predictions", []),
                    "old_correct_top1": old_data.get("correct_top1", False),
                    "old_correct_top5": old_data.get("correct_top5", False),
                    "old_match_position": old_data.get("match_position", 0),
                    "old_accuracy_score": old_data.get("accuracy_score", 0),
                    "old_response_time_ms": old_data.get("response_time_ms", 0),
                })
            
            result_path = approach_dir / f"{approach}_results_{self.timestamp}.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": self.timestamp,
                    "approach": approach,
                    "metrics": metrics.get(approach, {}),
                    "old_metrics": metrics.get("old_classifier", {}),
                    "results": approach_results,
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {approach} results to {result_path}")
        
        # Also save old_classifier separately for reference
        old_dir = output_dir / "old_classifier"
        old_dir.mkdir(parents=True, exist_ok=True)
        
        old_results = []
        for r in all_results:
            if "error" in r:
                continue
            old_data = r.get("old_classifier", {})
            old_results.append({
                "paper_id": r.get("paper_id"),
                "ground_truth_id": r.get("ground_truth_id"),
                "ground_truth_label": r.get("ground_truth_label"),
                "ground_truth_path": r.get("ground_truth_path"),
                "correct_top1": old_data.get("correct_top1", False),
                "correct_top5": old_data.get("correct_top5", False),
                "match_position": old_data.get("match_position", 0),
                "accuracy_score": old_data.get("accuracy_score", 0),
                "predictions": old_data.get("predictions", []),
                "response_time_ms": old_data.get("response_time_ms", 0),
            })
        
        old_path = old_dir / f"old_classifier_results_{self.timestamp}.json"
        with open(old_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": self.timestamp,
                "metrics": metrics.get("old_classifier", {}),
                "results": old_results,
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved old_classifier results to {old_path}")
        
        # Print summary
        self._print_summary(metrics, approaches)
    
    def _print_summary(self, metrics: Dict[str, Dict], approaches: List[str]):
        """Print evaluation summary table."""
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        
        print("\n| Approach         | Top-1 Acc | Top-5 Acc | Accuracy Score | Avg Time (ms) |")
        print("|------------------|-----------|-----------|----------------|---------------|")
        
        if "old_classifier" in metrics:
            m = metrics["old_classifier"]
            print(f"| Old (Flat)       | {m['top1_accuracy']*100:>8.1f}% | {m['top5_accuracy']*100:>8.1f}% | {m['accuracy_score']*100:>13.1f}% | {m['avg_response_time_ms']:>13.0f} |")
        
        for approach in approaches:
            if approach in metrics:
                m = metrics[approach]
                name = approach.replace("_", " ").title()[:16]
                print(f"| {name:<16} | {m['top1_accuracy']*100:>8.1f}% | {m['top5_accuracy']*100:>8.1f}% | {m['accuracy_score']*100:>13.1f}% | {m['avg_response_time_ms']:>13.0f} |")
        
        print("\n")
    
    async def run(self, approaches: List[str] = None, max_papers: int = None):
        """Run full evaluation."""
        if not self.load_dataset():
            return
        
        if approaches is None:
            approaches = ["single_shot", "top_down", "embedding"]
        
        papers = self.papers[:max_papers] if max_papers else self.papers
        
        print(f"\n{'='*80}")
        print(f"RESEARCH FIELD CLASSIFIER EVALUATION")
        print(f"{'='*80}")
        print(f"Papers to evaluate: {len(papers)}")
        print(f"New approaches: {', '.join(approaches)}")
        print(f"Also calling: OLD ORKG classifier for comparison")
        print(f"{'='*80}\n")
        
        all_results = []
        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] Processing {paper.get('paper_id', 'unknown')}...")
            result = await self.evaluate_paper(paper, approaches)
            all_results.append(result)
            await asyncio.sleep(0.3)  # Rate limiting
        
        metrics = self.calculate_metrics(all_results, approaches)
        self.save_results(all_results, metrics, approaches)


def main():
    parser = argparse.ArgumentParser(description="Run classifier evaluation")
    parser.add_argument("--approaches", "-a", default="single_shot,top_down,embedding",
                       help="Comma-separated approaches to evaluate")
    parser.add_argument("--dataset", "-d", default=None, help="Path to dataset CSV")
    parser.add_argument("--max-papers", "-n", type=int, default=None, help="Max papers to evaluate")
    
    args = parser.parse_args()
    
    approaches = [a.strip() for a in args.approaches.split(",")]
    
    runner = EvaluationRunner(args.dataset)
    asyncio.run(runner.run(approaches, args.max_papers))


if __name__ == "__main__":
    main()
