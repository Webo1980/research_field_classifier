"""
LLM Service (Base)
==================
Handles LLM API calls for all classifier approaches.
Supports OpenAI, Mistral, and Anthropic.
"""

import logging
from typing import Dict, List, Tuple, Optional
import httpx

from config import settings

logger = logging.getLogger(__name__)


class BaseLLMService:
    """
    Base LLM service for making API calls.
    Shared across all classifier approaches.
    """
    
    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self.provider_config: Optional[Dict] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the LLM service."""
        if self._initialized:
            return
        
        self.http_client = httpx.AsyncClient(timeout=180.0)
        self._setup_provider()
        self._initialized = True
        logger.info(f"LLM Service initialized: {settings.LLM_PROVIDER}")
    
    def _setup_provider(self):
        """Setup provider configuration based on settings."""
        provider = settings.LLM_PROVIDER.lower()
        
        if provider == "openai":
            self.provider_config = {
                "type": "openai",
                "api_key": settings.OPENAI_API_KEY,
                "model": settings.OPENAI_MODEL,
                "base_url": "https://api.openai.com/v1/chat/completions"
            }
        elif provider == "mistral":
            self.provider_config = {
                "type": "mistral",
                "api_key": settings.MISTRAL_API_KEY,
                "model": settings.MISTRAL_MODEL,
                "base_url": "https://api.mistral.ai/v1/chat/completions"
            }
        elif provider == "anthropic":
            self.provider_config = {
                "type": "anthropic",
                "api_key": settings.ANTHROPIC_API_KEY,
                "model": settings.ANTHROPIC_MODEL,
                "base_url": "https://api.anthropic.com/v1/messages"
            }
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self._initialized = False
    
    async def generate(
        self, 
        messages: List[Dict], 
        temperature: float = 0.0,
        max_tokens: int = 500
    ) -> Tuple[str, Dict]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        if not self._initialized:
            await self.initialize()
        
        provider = self.provider_config
        
        if provider["type"] in ["openai", "mistral"]:
            return await self._call_openai_style(messages, temperature, max_tokens)
        elif provider["type"] == "anthropic":
            return await self._call_anthropic(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider['type']}")
    
    async def _call_openai_style(
        self, 
        messages: List[Dict], 
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, Dict]:
        """Call OpenAI-style API (OpenAI, Mistral)."""
        provider = self.provider_config
        
        headers = {
            "Authorization": f"Bearer {provider['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": provider["model"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self.http_client.post(
            provider["base_url"],
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        return content, {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    
    async def _call_anthropic(
        self, 
        messages: List[Dict], 
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, Dict]:
        """Call Anthropic API."""
        provider = self.provider_config
        
        headers = {
            "x-api-key": provider["api_key"],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Extract system message
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        
        payload = {
            "model": provider["model"],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_msg,
            "messages": user_messages
        }
        
        response = await self.http_client.post(
            provider["base_url"],
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            response.raise_for_status()
        
        data = response.json()
        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        
        return content, {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        }


# Global singleton
llm_service = BaseLLMService()
