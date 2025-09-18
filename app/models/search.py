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
    vector_query: Optional[str] = Field(default=None, description="Override for the vector search query")
    business_query: Optional[str] = Field(default=None, description="Business brief for stage-two LLM scoring")
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
    
    # Post-filtering controls
    business_fit_query: Optional[str] = Field(default=None, description="Optional business brief for LLM-based post filtering")
    post_filter_limit: Optional[int] = Field(default=None, ge=1, description="Number of top results to re-score")
    post_filter_concurrency: Optional[int] = Field(default=8, ge=1, description="Concurrent LLM requests for post-filtering")
    post_filter_max_posts: Optional[int] = Field(default=6, ge=1, description="Recent posts per profile to include in prompt")
    post_filter_model: Optional[str] = Field(default="gpt-5-mini", description="OpenAI model for profile fit scoring")
    post_filter_verbosity: Optional[str] = Field(default="medium", description="Responses API verbosity")
    post_filter_use_brightdata: Optional[bool] = Field(default=False, description="Refresh top results via BrightData before scoring (blocking)")
    
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
    business_query: Optional[str] = None
    method: str
    debug: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ImageRefreshRequest(BaseModel):
    """Image refresh request"""
    usernames: List[str] = Field(..., min_items=1, max_items=50, description="List of usernames to refresh")
    update_database: bool = Field(default=False, description="Whether to update database with fresh data")


class ImageRefreshSearchRequest(BaseModel):
    """Image refresh for search results request"""
    search_results: List[Dict[str, Any]] = Field(..., min_items=1, description="Search results to refresh images for")
    update_database: bool = Field(default=True, description="Whether to update database with fresh data")


class ProfileFitTestRequest(BaseModel):
    """Request payload to score a single profile against a business brief."""

    business_fit_query: str = Field(..., description="Business goals and desired creator traits")
    account: Optional[str] = Field(default=None, description="Existing username to pull from LanceDB")
    profile_url: Optional[str] = Field(default=None, description="Profile URL (used if account unavailable)")
    max_posts: int = Field(default=6, ge=1, le=20, description="Maximum recent posts to include")
    model: str = Field(default="gpt-5-mini", description="OpenAI model for scoring")
    verbosity: str = Field(default="medium", description="Responses API verbosity mode")
    use_brightdata: bool = Field(default=False, description="Refresh profile data via BrightData before scoring")
    concurrency: int = Field(default=2, ge=1, le=8, description="Concurrent calls (use >1 when use_brightdata is false)")
