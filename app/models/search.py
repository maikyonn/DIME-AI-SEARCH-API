"""Search-related Pydantic models"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class CustomWeights(BaseModel):
    """Vector search weights"""
    keyword: float = Field(ge=0.0, le=1.0, description="Keyword vector weight")
    profile: float = Field(ge=0.0, le=1.0, description="Profile vector weight")
    content: float = Field(ge=0.0, le=1.0, description="Content vector weight")


class SearchRequest(BaseModel):
    """Main search request model with maximum customizability"""
    query: str = Field(..., min_length=1, description="Search query for creators")
    method: str = Field(default="hybrid", description="Search method: vector, text, or hybrid")
    limit: int = Field(default=20, ge=1, description="Maximum number of results (no upper limit)")
    
    # Follower Range Filters
    min_followers: int = Field(default=0, ge=0, description="Minimum follower count")
    max_followers: Optional[int] = Field(default=None, ge=0, description="Maximum follower count (optional)")
    
    # Engagement Filters
    min_engagement: float = Field(default=0.0, ge=0.0, description="Minimum engagement rate")
    max_engagement: Optional[float] = Field(default=None, ge=0.0, description="Maximum engagement rate (optional)")
    
    # Content and Location Filters
    location: Optional[str] = Field(default=None, description="Location filter")
    category: Optional[str] = Field(default=None, description="Business category filter")
    keywords: Optional[List[str]] = Field(default=None, description="Additional keywords")
    
    # Advanced Search Controls
    custom_weights: Optional[CustomWeights] = Field(default=None, description="Custom vector search weights")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    return_vectors: bool = Field(default=False, description="Include vector data in results")
    
    # Account Type Filters
    is_verified: Optional[bool] = Field(default=None, description="Filter by verified status")
    is_business_account: Optional[bool] = Field(default=None, description="Filter by business account status")
    
    # LLM Score Filters (with full range control)
    min_individual_vs_org_score: Optional[int] = Field(default=None, ge=0, le=10, description="Minimum individual vs org score")
    max_individual_vs_org_score: Optional[int] = Field(default=None, ge=0, le=10, description="Maximum individual vs org score")
    min_generational_appeal_score: Optional[int] = Field(default=None, ge=0, le=10, description="Minimum generational appeal score")
    max_generational_appeal_score: Optional[int] = Field(default=None, ge=0, le=10, description="Maximum generational appeal score")
    min_professionalization_score: Optional[int] = Field(default=None, ge=0, le=10, description="Minimum professionalization score")
    max_professionalization_score: Optional[int] = Field(default=None, ge=0, le=10, description="Maximum professionalization score")
    min_relationship_status_score: Optional[int] = Field(default=None, ge=0, le=10, description="Minimum relationship status score")
    max_relationship_status_score: Optional[int] = Field(default=None, ge=0, le=10, description="Maximum relationship status score")
    
    # Content Stats Filters
    min_posts_count: Optional[int] = Field(default=None, ge=0, description="Minimum posts count")
    max_posts_count: Optional[int] = Field(default=None, ge=0, description="Maximum posts count")
    min_following: Optional[int] = Field(default=None, ge=0, description="Minimum following count")
    max_following: Optional[int] = Field(default=None, ge=0, description="Maximum following count")


class SimilarSearchRequest(BaseModel):
    """Similar creator search request with enhanced customization"""
    account: str = Field(..., min_length=1, description="Reference account username")
    limit: int = Field(default=10, ge=1, description="Maximum number of results (no upper limit)")
    
    # Follower Range Filters
    min_followers: int = Field(default=0, ge=0, description="Minimum follower count")
    max_followers: Optional[int] = Field(default=None, ge=0, description="Maximum follower count (optional)")
    
    # Advanced Search Controls
    similarity_threshold: Optional[float] = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
    use_vector_similarity: bool = Field(default=True, description="Whether to use vector similarity search")
    custom_weights: Optional[CustomWeights] = Field(default=None, description="Custom vector similarity weights")
    
    # Additional Filters
    min_engagement: Optional[float] = Field(default=None, ge=0.0, description="Minimum engagement rate")
    max_engagement: Optional[float] = Field(default=None, ge=0.0, description="Maximum engagement rate")
    location: Optional[str] = Field(default=None, description="Location filter")
    category: Optional[str] = Field(default=None, description="Business category filter")


class CategorySearchRequest(BaseModel):
    """Category search request with full customization"""
    category: str = Field(..., min_length=1, description="Category to search for")
    location: Optional[str] = Field(default=None, description="Location filter")
    limit: int = Field(default=15, ge=1, description="Maximum number of results (no upper limit)")
    
    # Follower Range Filters
    min_followers: int = Field(default=0, ge=0, description="Minimum follower count")
    max_followers: Optional[int] = Field(default=None, ge=0, description="Maximum follower count (optional)")
    
    # Engagement Filters
    min_engagement: Optional[float] = Field(default=None, ge=0.0, description="Minimum engagement rate")
    max_engagement: Optional[float] = Field(default=None, ge=0.0, description="Maximum engagement rate")
    
    # Advanced Controls
    custom_weights: Optional[CustomWeights] = Field(default=None, description="Custom vector search weights")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum similarity threshold")


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