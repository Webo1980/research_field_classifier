"""
Configuration Settings
======================
Central configuration for all classifier approaches.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    DEBUG: bool = False
    
    # LLM Provider Settings
    LLM_PROVIDER: str = "openai"  # openai, mistral, anthropic
    
    # OpenAI Settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Mistral Settings
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "mistral-large-latest"
    
    # Anthropic Settings
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
    
    # Embedding Settings (for embedding approach)
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # ORKG Settings
    ORKG_API_URL: str = "https://orkg.org/api"
    ORKG_ROOT_FIELD_ID: str = "R11"
    
    # Cache Settings
    CACHE_DIR: str = "./cache"
    TAXONOMY_CACHE_FILE: str = "taxonomy_hierarchy.json"
    TAXONOMY_TREE_CACHE_FILE: str = "taxonomy_tree.txt"
    TAXONOMY_EMBEDDINGS_CACHE_FILE: str = "taxonomy_embeddings.npz"
    TAXONOMY_CACHE_TTL_HOURS: int = 168  # 1 week
    EMBEDDINGS_CACHE_TTL_HOURS: int = 168  # 1 week (should match taxonomy TTL)
    
    # Evaluation Settings
    EVALUATION_RESULTS_DIR: str = "./evaluation_results"
    REPORTS_DIR: str = "./reports"
    
    # Approach Settings
    DEFAULT_APPROACH: str = "single_shot"  # single_shot, top_down, embedding
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
