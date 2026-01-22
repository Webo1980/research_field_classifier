"""
Taxonomy Service (Base)
=======================
Manages the ORKG research field taxonomy.
Shared across all classifier approaches.

Features:
- Dynamic fetching from ORKG API
- Local caching with TTL
- Automatic refresh when cache expires
"""

import json
import os
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class TaxonomyService:
    """
    Service for managing the ORKG research field taxonomy.
    
    Features:
    - Fetches taxonomy from ORKG API on startup
    - Caches locally for performance
    - Auto-refreshes when cache expires (configurable TTL)
    """
    
    def __init__(self):
        self._hierarchy: Optional[Dict] = None
        self._compact_tree: Optional[str] = None
        self._cache_loaded_at: Optional[datetime] = None
        
        # Indexes for fast lookup
        self._node_index: Dict[str, Dict] = {}
        self._label_index: Dict[str, Dict] = {}
        self._path_index: Dict[str, List[str]] = {}
        self._path_ids_index: Dict[str, List[str]] = {}
        self._parent_index: Dict[str, str] = {}
        self._depth_index: Dict[str, int] = {}
        self._leaf_nodes: List[str] = []
        
        os.makedirs(settings.CACHE_DIR, exist_ok=True)
    
    @property
    def hierarchy_cache_path(self) -> Path:
        return Path(settings.CACHE_DIR) / settings.TAXONOMY_CACHE_FILE
    
    @property
    def tree_cache_path(self) -> Path:
        return Path(settings.CACHE_DIR) / settings.TAXONOMY_TREE_CACHE_FILE
    
    def _is_cache_valid(self) -> bool:
        """Check if cached taxonomy is still valid based on TTL."""
        if not self.hierarchy_cache_path.exists():
            return False
        
        try:
            mtime = datetime.fromtimestamp(self.hierarchy_cache_path.stat().st_mtime)
            ttl = timedelta(hours=settings.TAXONOMY_CACHE_TTL_HOURS)
            is_valid = datetime.now() - mtime < ttl
            
            if not is_valid:
                logger.info(f"Cache expired (TTL: {settings.TAXONOMY_CACHE_TTL_HOURS}h)")
            
            return is_valid
        except Exception:
            return False
    
    async def initialize(self, force_refresh: bool = False):
        """
        Initialize the taxonomy service.
        
        Args:
            force_refresh: If True, fetch fresh data from ORKG API
        """
        # Check if we need to fetch from API
        should_fetch = force_refresh or not self._is_cache_valid()
        
        if should_fetch:
            logger.info("Fetching taxonomy from ORKG API...")
            await self._fetch_from_orkg_api()
        elif self.tree_cache_path.exists():
            logger.info("Loading taxonomy from cached tree file...")
            if self._load_from_tree_file():
                logger.info(f"Taxonomy initialized: {len(self._node_index)} fields")
                return
        elif self.hierarchy_cache_path.exists():
            logger.info("Loading taxonomy from hierarchy cache...")
            self._load_from_hierarchy_cache()
            self._build_indexes()
            self._generate_compact_tree()
        else:
            # No cache exists, must fetch
            logger.info("No cache found. Fetching taxonomy from ORKG API...")
            await self._fetch_from_orkg_api()
        
        logger.info(f"Taxonomy initialized: {len(self._node_index)} fields, {len(self._leaf_nodes)} leaf nodes")
    
    async def _fetch_from_orkg_api(self):
        """Fetch complete taxonomy from ORKG API and cache it."""
        from services.base.orkg_api_service import orkg_api_service
        
        try:
            # Fetch complete taxonomy tree
            root_id = settings.ORKG_ROOT_FIELD_ID
            self._hierarchy = orkg_api_service.fetch_complete_taxonomy(root_id)
            
            # Save to cache
            with open(self.hierarchy_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._hierarchy, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved taxonomy hierarchy to {self.hierarchy_cache_path}")
            
            # Build indexes and tree
            self._build_indexes()
            self._generate_compact_tree()
            
            self._cache_loaded_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to fetch taxonomy from ORKG API: {e}")
            
            # Try to fall back to existing cache
            if self.tree_cache_path.exists():
                logger.info("Falling back to existing cache...")
                self._load_from_tree_file()
            elif self.hierarchy_cache_path.exists():
                logger.info("Falling back to hierarchy cache...")
                self._load_from_hierarchy_cache()
                self._build_indexes()
                self._generate_compact_tree()
            else:
                raise RuntimeError(
                    "Failed to fetch taxonomy and no cache available. "
                    "Please check your internet connection and ORKG API availability."
                )
    
    def _load_from_tree_file(self) -> bool:
        """Load taxonomy from compact tree file."""
        try:
            with open(self.tree_cache_path, 'r', encoding='utf-8') as f:
                self._compact_tree = f.read()
            
            self._node_index = {}
            self._label_index = {}
            self._path_index = {}
            self._path_ids_index = {}
            self._parent_index = {}
            self._depth_index = {}
            self._leaf_nodes = []
            
            for line in self._compact_tree.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(r'\[([^\]]+)\]\s*(.+)', line)
                if not match:
                    continue
                
                field_id = match.group(1)
                path_str = match.group(2)
                path_labels = [p.strip() for p in path_str.split('>')]
                leaf_label = path_labels[-1] if path_labels else ""
                
                node = {"id": field_id, "label": leaf_label, "children": []}
                self._node_index[field_id] = node
                self._label_index[leaf_label.lower()] = node
                
                full_path = ["Research Field"] + path_labels
                self._path_index[field_id] = full_path
                self._path_ids_index[field_id] = [field_id]
                self._depth_index[field_id] = len(path_labels)
                self._leaf_nodes.append(field_id)
            
            return len(self._leaf_nodes) > 0
            
        except Exception as e:
            logger.error(f"Failed to load from tree file: {e}")
            return False
    
    def _load_from_hierarchy_cache(self):
        """Load taxonomy from hierarchy JSON cache."""
        try:
            with open(self.hierarchy_cache_path, 'r', encoding='utf-8') as f:
                self._hierarchy = json.load(f)
            self._cache_loaded_at = datetime.now()
        except Exception as e:
            logger.error(f"Failed to load hierarchy cache: {e}")
            raise
    
    def _build_indexes(self):
        """Build indexes from hierarchy."""
        if not self._hierarchy:
            return
        
        self._node_index = {}
        self._label_index = {}
        self._path_index = {}
        self._path_ids_index = {}
        self._parent_index = {}
        self._depth_index = {}
        self._leaf_nodes = []
        
        def index_node(node: Dict, parent_id: Optional[str], path_labels: List[str], path_ids: List[str], depth: int):
            node_id = node.get("id", "")
            label = node.get("label", "")
            children = node.get("children", [])
            
            self._node_index[node_id] = node
            self._label_index[label.lower()] = node
            
            current_path_labels = path_labels + [label]
            current_path_ids = path_ids + [node_id]
            self._path_index[node_id] = current_path_labels
            self._path_ids_index[node_id] = current_path_ids
            
            if parent_id:
                self._parent_index[node_id] = parent_id
            
            self._depth_index[node_id] = depth
            
            if not children:
                self._leaf_nodes.append(node_id)
            
            for child in children:
                index_node(child, node_id, current_path_labels, current_path_ids, depth + 1)
        
        index_node(self._hierarchy, None, [], [], 0)
    
    def _generate_compact_tree(self):
        """Generate compact tree representation."""
        lines = []
        
        for leaf_id in self._leaf_nodes:
            path_labels = self._path_index.get(leaf_id, [])
            if path_labels:
                display_path = path_labels[1:] if len(path_labels) > 1 else path_labels
                path_str = " > ".join(display_path)
                lines.append(f"[{leaf_id}] {path_str}")
        
        lines.sort()
        self._compact_tree = "\n".join(lines)
        
        try:
            with open(self.tree_cache_path, 'w', encoding='utf-8') as f:
                f.write(self._compact_tree)
        except Exception as e:
            logger.error(f"Failed to save compact tree: {e}")
    
    def get_compact_tree(self) -> str:
        """Get compact tree representation."""
        return self._compact_tree or ""
    
    def get_node_by_id(self, field_id: str) -> Optional[Dict]:
        """Get node by ID."""
        return self._node_index.get(field_id)
    
    def get_node_by_label(self, label: str) -> Optional[Dict]:
        """Get node by label (case-insensitive)."""
        return self._label_index.get(label.lower())
    
    def get_path_labels(self, field_id: str) -> List[str]:
        """Get full path labels for a field."""
        return self._path_index.get(field_id, [])
    
    def get_path_ids(self, field_id: str) -> List[str]:
        """Get full path IDs for a field."""
        return self._path_ids_index.get(field_id, [])
    
    def get_parent_id(self, field_id: str) -> Optional[str]:
        """Get parent ID of a field."""
        return self._parent_index.get(field_id)
    
    def get_depth(self, field_id: str) -> int:
        """Get depth of a field."""
        return self._depth_index.get(field_id, 0)
    
    def is_leaf(self, field_id: str) -> bool:
        """Check if field is a leaf node."""
        return field_id in self._leaf_nodes
    
    def get_all_leaf_ids(self) -> List[str]:
        """Get all leaf node IDs."""
        return self._leaf_nodes.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get taxonomy statistics."""
        depths = list(self._depth_index.values())
        return {
            "total_nodes": len(self._node_index),
            "leaf_nodes": len(self._leaf_nodes),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "compact_tree_tokens": len(self._compact_tree) // 4 if self._compact_tree else 0
        }


# Global singleton
taxonomy_service = TaxonomyService()
