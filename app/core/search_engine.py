"""
FastAPI wrapper for the core search engine.
This module provides a clean interface to the existing search functionality.
"""
import os
import sys
from typing import List, Optional, Dict, Any

# Add the original src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(os.path.join(project_root, "src"))

# Import from the original search engine
from core.search_engine import GenZCreatorSearchEngine as OriginalSearchEngine, SearchResult


class FastAPISearchEngine:
    """FastAPI wrapper for the GenZCreatorSearchEngine"""
    
    def __init__(self, db_path: str):
        self.engine = OriginalSearchEngine(db_path)
    
    def search_creators_for_campaign(
        self,
        query: str,
        method: str = "hybrid",
        limit: int = 20,
        min_followers: int = 1000,
        max_followers: int = 10000000,
        min_engagement: float = 0.0,
        location_filter: Optional[str] = None,
        target_category: Optional[str] = None,
        relevance_keywords: Optional[List[str]] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """Search creators for a campaign"""
        return self.engine.search_creators_for_campaign(
            query=query,
            method=method,
            limit=limit,
            min_followers=min_followers,
            max_followers=max_followers,
            min_engagement=min_engagement,
            location_filter=location_filter,
            target_category=target_category,
            relevance_keywords=relevance_keywords,
            custom_weights=custom_weights
        )
    
    def match_creators_to_business(
        self,
        business_description: str,
        method: str = "hybrid",
        limit: int = 20,
        custom_weights: Optional[Dict[str, float]] = None,
        min_followers: int = 1000,
        max_followers: int = 10000000,
        min_engagement: float = 0.0,
        location_filter: Optional[str] = None,
        target_category: Optional[str] = None
    ) -> List[SearchResult]:
        """Match creators to business description"""
        return self.engine.match_creators_to_business(
            business_description=business_description,
            method=method,
            limit=limit,
            custom_weights=custom_weights,
            min_followers=min_followers,
            max_followers=max_followers,
            min_engagement=min_engagement,
            location_filter=location_filter,
            target_category=target_category
        )
    
    def find_similar_creators(
        self,
        reference_account: str,
        limit: int = 10,
        min_followers: int = 1000
    ) -> List[SearchResult]:
        """Find similar creators to a reference account"""
        return self.engine.find_similar_creators(
            reference_account=reference_account,
            limit=limit,
            min_followers=min_followers
        )
    
    def search_by_category(
        self,
        category: str,
        location: Optional[str] = None,
        min_followers: int = 5000,
        limit: int = 15
    ) -> List[SearchResult]:
        """Search creators by category"""
        return self.engine.search_by_category(
            category=category,
            location=location,
            min_followers=min_followers,
            limit=limit
        )