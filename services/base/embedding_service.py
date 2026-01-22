"""
Embedding Service (Base)
========================
Handles embedding generation via OpenAI API.
Used by the embedding-based classifier approach.

Features:
- Dynamic embedding generation from ORKG taxonomy
- TTL-based cache validation
- Automatic regeneration when taxonomy changes
- Metadata tracking for cache freshness
"""

import logging
import asyncio
import hashlib
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for computing text embeddings via OpenAI API.
    """
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.api_key = settings.OPENAI_API_KEY
        self.http_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the embedding service."""
        if self._initialized:
            return
        
        import httpx
        self.http_client = httpx.AsyncClient(timeout=60.0)
        self._initialized = True
        logger.info(f"Embedding service initialized: {self.model_name}")
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self._initialized = False
    
    async def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of embeddings
        """
        if not self._initialized:
            await self.initialize()
        
        # Truncate texts to avoid token limits
        truncated_texts = [t[:8000] for t in texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": truncated_texts
        }
        
        response = await self.http_client.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Embedding API error: {response.status_code}")
            response.raise_for_status()
        
        data = response.json()
        
        # Extract embeddings in order
        embeddings = [None] * len(texts)
        for item in data["data"]:
            embeddings[item["index"]] = np.array(item["embedding"])
        
        return np.array(embeddings)
    
    async def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        embeddings = await self.encode([text])
        return embeddings[0]


class TaxonomyEmbeddings:
    """
    Pre-computed embeddings for taxonomy fields with dynamic generation and caching.
    
    Features:
    - Fetches taxonomy from ORKG API dynamically (via TaxonomyService)
    - Computes embeddings for all research fields
    - Caches embeddings with TTL-based validation
    - Automatically regenerates when taxonomy changes
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.field_embeddings: Dict[str, np.ndarray] = {}
        self.field_info: Dict[str, Dict] = {}
        self.field_by_name: Dict[str, str] = {}
        self._initialized = False
        self._stats = {}
        self._cache_metadata: Dict = {}
    
    @property
    def cache_path(self) -> Path:
        """Path to the embeddings cache file."""
        return Path(settings.CACHE_DIR) / settings.TAXONOMY_EMBEDDINGS_CACHE_FILE
    
    @property
    def metadata_path(self) -> Path:
        """Path to the cache metadata file."""
        return Path(settings.CACHE_DIR) / "taxonomy_embeddings_meta.json"
    
    def _compute_taxonomy_hash(self, field_ids: List[str]) -> str:
        """Compute a hash of taxonomy field IDs to detect changes."""
        content = "|".join(sorted(field_ids))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, current_taxonomy_hash: str) -> bool:
        """
        Check if cached embeddings are still valid.
        
        Validation checks:
        1. Cache file exists
        2. Metadata file exists with timestamp
        3. Cache is within TTL
        4. Taxonomy hash matches (no fields added/removed)
        5. Embedding model matches
        """
        if not self.cache_path.exists() or not self.metadata_path.exists():
            logger.info("Embeddings cache or metadata not found")
            return False
        
        try:
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check TTL
            cached_at = datetime.fromisoformat(metadata.get("created_at", "1970-01-01"))
            ttl = timedelta(hours=settings.EMBEDDINGS_CACHE_TTL_HOURS)
            if datetime.now() - cached_at > ttl:
                logger.info(f"Embeddings cache expired (TTL: {settings.EMBEDDINGS_CACHE_TTL_HOURS}h)")
                return False
            
            # Check taxonomy hash
            cached_hash = metadata.get("taxonomy_hash", "")
            if cached_hash != current_taxonomy_hash:
                logger.info("Taxonomy changed - embeddings cache invalidated")
                logger.debug(f"Cached hash: {cached_hash}, Current hash: {current_taxonomy_hash}")
                return False
            
            # Check embedding model
            cached_model = metadata.get("embedding_model", "")
            if cached_model != settings.EMBEDDING_MODEL:
                logger.info(f"Embedding model changed ({cached_model} -> {settings.EMBEDDING_MODEL})")
                return False
            
            logger.info("Embeddings cache is valid")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to validate cache metadata: {e}")
            return False
    
    def _save_metadata(self, taxonomy_hash: str, field_count: int):
        """Save cache metadata for validation."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "taxonomy_hash": taxonomy_hash,
            "embedding_model": settings.EMBEDDING_MODEL,
            "field_count": field_count,
            "ttl_hours": settings.EMBEDDINGS_CACHE_TTL_HOURS
        }
        
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved embeddings metadata to {self.metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")
    
    async def initialize(self, taxonomy_tree_path: Path, force_refresh: bool = False):
        """
        Load taxonomy and compute/load embeddings.
        
        Args:
            taxonomy_tree_path: Path to taxonomy_tree.txt file
            force_refresh: If True, regenerate embeddings even if cache is valid
        """
        if self._initialized and not force_refresh:
            return
        
        import re
        
        # Ensure cache directory exists
        Path(settings.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        # Load taxonomy from tree file
        logger.info(f"Loading taxonomy from {taxonomy_tree_path}...")
        
        if not taxonomy_tree_path.exists():
            raise FileNotFoundError(
                f"Taxonomy tree file not found: {taxonomy_tree_path}. "
                "Ensure TaxonomyService has been initialized first."
            )
        
        with open(taxonomy_tree_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse taxonomy
        field_texts = []
        field_ids = []
        depths = []
        
        self.field_info = {}
        self.field_by_name = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(r'\[([^\]]+)\]\s*(.+)', line)
            if not match:
                continue
            
            field_id = match.group(1)
            path_str = match.group(2)
            parts = [p.strip() for p in path_str.split('>')]
            leaf_name = parts[-1]
            
            self.field_info[field_id] = {
                'id': field_id,
                'label': leaf_name,
                'path': parts,
                'path_str': path_str
            }
            
            # Build name lookup
            self.field_by_name[leaf_name.lower()] = field_id
            
            # Build embedding text (includes path for context)
            field_text = f"{path_str}. {leaf_name}"
            field_texts.append(field_text)
            field_ids.append(field_id)
            depths.append(len(parts))
        
        logger.info(f"Parsed {len(field_ids)} research fields from taxonomy")
        
        # Compute taxonomy hash for cache validation
        taxonomy_hash = self._compute_taxonomy_hash(field_ids)
        
        # Check if we can use cached embeddings
        should_regenerate = force_refresh or not self._is_cache_valid(taxonomy_hash)
        
        if not should_regenerate:
            # Try to load from cache
            try:
                logger.info(f"Loading cached embeddings from {self.cache_path}")
                cached = np.load(self.cache_path, allow_pickle=True)
                cached_ids = cached['field_ids'].tolist()
                cached_embeddings = cached['embeddings']
                
                # Verify IDs match exactly (should always pass if hash matched)
                if cached_ids == field_ids:
                    for fid, emb in zip(cached_ids, cached_embeddings):
                        self.field_embeddings[fid] = emb
                    logger.info(f"Loaded {len(self.field_embeddings)} embeddings from cache")
                else:
                    logger.warning("Cache ID mismatch despite hash match - regenerating")
                    should_regenerate = True
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
                should_regenerate = True
        
        # Generate embeddings if needed
        if should_regenerate or not self.field_embeddings:
            logger.info(f"Generating embeddings for {len(field_texts)} research fields...")
            logger.info(f"This may take a few minutes on first run...")
            
            await self._generate_embeddings(field_texts, field_ids)
            
            # Save cache and metadata
            self._save_cache(field_ids)
            self._save_metadata(taxonomy_hash, len(field_ids))
        
        # Compute statistics
        self._stats = {
            "total_nodes": len(field_ids),
            "leaf_nodes": len(field_ids),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "embedding_dimension": len(next(iter(self.field_embeddings.values()))) if self.field_embeddings else 0,
            "cache_valid": not should_regenerate
        }
        
        self._initialized = True
        logger.info(f"Taxonomy embeddings initialized: {len(self.field_embeddings)} fields")
    
    async def _generate_embeddings(self, field_texts: List[str], field_ids: List[str]):
        """
        Generate embeddings for all taxonomy fields in batches.
        
        Args:
            field_texts: List of field text descriptions
            field_ids: List of corresponding field IDs
        """
        batch_size = 100
        total_batches = (len(field_texts) + batch_size - 1) // batch_size
        
        self.field_embeddings = {}
        
        for i in range(0, len(field_texts), batch_size):
            batch_num = i // batch_size + 1
            batch = field_texts[i:i + batch_size]
            batch_ids = field_ids[i:i + batch_size]
            
            logger.info(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} fields)...")
            
            try:
                embeddings = await self.embedding_service.encode(batch)
                
                for fid, emb in zip(batch_ids, embeddings):
                    self.field_embeddings[fid] = emb
                
                # Rate limiting - avoid hitting API limits
                if batch_num < total_batches:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {batch_num}: {e}")
                raise
        
        logger.info(f"Generated {len(self.field_embeddings)} embeddings")
    
    def _save_cache(self, field_ids: List[str]):
        """Save embeddings to cache file."""
        try:
            embeddings_array = np.array([self.field_embeddings[fid] for fid in field_ids])
            
            np.savez(
                self.cache_path,
                field_ids=np.array(field_ids),
                embeddings=embeddings_array
            )
            logger.info(f"Saved embeddings cache to {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")
    
    @property
    def stats(self) -> Dict:
        """Get statistics about the taxonomy embeddings."""
        return self._stats
    
    def get_field_by_name(self, name: str) -> Optional[str]:
        """Get field ID by name (case-insensitive)."""
        return self.field_by_name.get(name.lower())
    
    def get_field_info(self, field_id: str) -> Optional[Dict]:
        """Get field information by ID."""
        return self.field_info.get(field_id)
    
    def find_similar(self, query_embedding: np.ndarray, top_n: int = 5) -> List:
        """
        Find most similar research fields to a query embedding.
        
        Args:
            query_embedding: Query vector
            top_n: Number of results to return
            
        Returns:
            List of (field_id, similarity_score, field_info) tuples
        """
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        
        for field_id, field_embedding in self.field_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, field_embedding) / (
                query_norm * np.linalg.norm(field_embedding)
            )
            similarities.append((field_id, float(similarity), self.field_info[field_id]))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_n]
    
    async def refresh(self, taxonomy_tree_path: Path):
        """
        Force refresh embeddings from current taxonomy.
        
        Use this when you know the taxonomy has been updated.
        """
        logger.info("Forcing embeddings refresh...")
        self._initialized = False
        self.field_embeddings = {}
        await self.initialize(taxonomy_tree_path, force_refresh=True)


# Global singletons
embedding_service = EmbeddingService()
