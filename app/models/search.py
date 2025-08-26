"""Search-related Pydantic models"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class CustomWeights(BaseModel):
    """Custom scoring weights for search"""
    business_alignment: float = Field(default=0.25, ge=0.0, le=1.0)
    genz_appeal: float = Field(default=0.20, ge=0.0, le=1.0)
    authenticity: float = Field(default=0.20, ge=0.0, le=1.0)
    engagement: float = Field(default=0.20, ge=0.0, le=1.0)
    campaign_value: float = Field(default=0.15, ge=0.0, le=1.0)

    @validator('*')
    def validate_sum(cls, v, values):
        """Ensure weights don't exceed reasonable bounds"""
        return v


class SearchRequest(BaseModel):
    """Main search request model"""
    query: str = Field(..., min_length=1, description="Business description or search query")
    is_business_description: bool = Field(default=False, description="Whether query is a business description")
    method: str = Field(default="hybrid", description="Search method: vector, text, or hybrid")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    min_followers: int = Field(default=1000, ge=0, description="Minimum follower count")
    max_followers: int = Field(default=10000000, ge=0, description="Maximum follower count")
    min_engagement: float = Field(default=0.0, ge=0.0, description="Minimum engagement rate")
    location: Optional[str] = Field(default=None, description="Location filter")
    category: Optional[str] = Field(default=None, description="Business category filter")
    keywords: Optional[List[str]] = Field(default=None, description="Additional keywords")
    weights: Optional[CustomWeights] = Field(default=None, description="Custom scoring weights")


class SimilarSearchRequest(BaseModel):
    """Similar creator search request"""
    account: str = Field(..., min_length=1, description="Reference account username")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    min_followers: int = Field(default=1000, ge=0, description="Minimum follower count")


class CategorySearchRequest(BaseModel):
    """Category search request"""
    category: str = Field(..., min_length=1, description="Category to search for")
    location: Optional[str] = Field(default=None, description="Location filter")
    limit: int = Field(default=15, ge=1, le=50, description="Maximum number of results")
    min_followers: int = Field(default=5000, ge=0, description="Minimum follower count")


class SearchResponse(BaseModel):
    """Search response model"""
    success: bool
    results: List[Dict[str, Any]]
    count: int
    query: str
    method: str
    error: Optional[str] = None


class ImageRefreshRequest(BaseModel):
    """Image refresh request"""
    usernames: List[str] = Field(..., min_items=1, max_items=50, description="List of usernames to refresh")
    update_database: bool = Field(default=False, description="Whether to update database with fresh data")


class ImageRefreshSearchRequest(BaseModel):
    """Image refresh for search results request"""
    search_results: List[Dict[str, Any]] = Field(..., min_items=1, description="Search results to refresh images for")
    update_database: bool = Field(default=True, description="Whether to update database with fresh data")