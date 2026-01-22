"""
ORKG API Service
================
Handles communication with the ORKG API for fetching taxonomy data.
"""

import logging
from typing import Dict, List, Optional, Any
import httpx

from config import settings

logger = logging.getLogger(__name__)


class ORKGAPIService:
    """
    Service for interacting with the ORKG API.
    Fetches research fields dynamically.
    """
    
    def __init__(self):
        self.base_url = settings.ORKG_API_URL
        self.http_client: Optional[httpx.Client] = None
        self._initialized = False
    
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if not self.http_client:
            self.http_client = httpx.Client(timeout=60.0)
        return self.http_client
    
    def close(self):
        """Close HTTP client."""
        if self.http_client:
            self.http_client.close()
            self.http_client = None
    
    def get_resource(self, resource_id: str) -> Optional[Dict]:
        """
        Get a single resource by ID.
        
        Args:
            resource_id: The ORKG resource ID (e.g., 'R11')
            
        Returns:
            Resource data or None if not found
        """
        try:
            client = self._get_client()
            url = f"{self.base_url}/resources/{resource_id}"
            response = client.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Resource {resource_id} not found: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch resource {resource_id}: {e}")
            return None
    
    def get_children(self, field_id: str, page: int = 0, size: int = 100) -> List[Dict]:
        """
        Get child fields of a parent field.
        
        Args:
            field_id: Parent field ID
            page: Page number for pagination
            size: Page size
            
        Returns:
            List of child field resources
        """
        try:
            client = self._get_client()
            url = f"{self.base_url}/statements"
            params = {
                "subject_id": field_id,
                "predicate_id": "P36",  # hasSubfield predicate
                "page": page,
                "size": size
            }
            
            response = client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", [])
                children = []
                for statement in content:
                    obj = statement.get("object", {})
                    if obj.get("id"):
                        children.append({
                            "id": obj.get("id"),
                            "label": obj.get("label", "Unknown")
                        })
                return children
            else:
                logger.warning(f"Failed to get children for {field_id}: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Failed to fetch children for {field_id}: {e}")
            return []
    
    def get_all_children(self, field_id: str) -> List[Dict]:
        """
        Get all children of a field (handles pagination).
        
        Args:
            field_id: Parent field ID
            
        Returns:
            List of all child field resources
        """
        all_children = []
        page = 0
        size = 100
        
        while True:
            children = self.get_children(field_id, page=page, size=size)
            if not children:
                break
            all_children.extend(children)
            if len(children) < size:
                break
            page += 1
        
        return all_children
    
    def fetch_complete_taxonomy(self, root_id: str = "R11", max_depth: int = 6) -> Dict:
        """
        Recursively fetch the complete taxonomy tree from ORKG.
        
        Args:
            root_id: Root field ID (default: R11 = "Research Field")
            max_depth: Maximum recursion depth
            
        Returns:
            Complete taxonomy tree as nested dict
        """
        logger.info(f"Fetching complete taxonomy from ORKG API (root: {root_id})...")
        
        def fetch_subtree(field_id: str, depth: int = 0) -> Dict:
            if depth > max_depth:
                logger.warning(f"Max depth {max_depth} reached at {field_id}")
                return {"id": field_id, "label": f"Depth limit ({field_id})", "children": []}
            
            # Get resource info
            resource = self.get_resource(field_id)
            if not resource:
                return {"id": field_id, "label": f"Unknown ({field_id})", "children": []}
            
            node = {
                "id": resource.get("id", field_id),
                "label": resource.get("label", "Unknown"),
                "children": []
            }
            
            # Get children
            children = self.get_all_children(field_id)
            
            indent = "  " * depth
            logger.debug(f"{indent}Field: {node['label']} ({len(children)} children)")
            
            # Recursively fetch subtrees
            for child in children:
                child_id = child.get("id")
                if child_id:
                    child_subtree = fetch_subtree(child_id, depth + 1)
                    node["children"].append(child_subtree)
            
            return node
        
        return fetch_subtree(root_id)


# Global singleton
orkg_api_service = ORKGAPIService()
