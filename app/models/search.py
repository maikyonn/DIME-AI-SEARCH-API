"""Search-related Pydantic models for the simplified API."""
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for creators")
    method: Literal["lexical", "semantic", "hybrid"] = Field(
        default="hybrid", description="Search mode"
    )
    limit: int = Field(default=20, ge=1, le=50000, description="Maximum results to return")

    min_followers: Optional[int] = Field(default=None, ge=0)
    max_followers: Optional[int] = Field(default=None, ge=0)
    min_engagement: Optional[float] = Field(default=None, ge=0.0)
    max_engagement: Optional[float] = Field(default=None, ge=0.0)

    location: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    is_verified: Optional[bool] = Field(default=None)
    is_business_account: Optional[bool] = Field(default=None)
    lexical_scope: Literal["bio", "bio_posts"] = Field(
        default="bio", description="Lexical search scope"
    )


class SimilarSearchRequest(BaseModel):
    account: str = Field(..., min_length=1, description="Reference account username")
    limit: int = Field(default=10, ge=1, le=100)

    min_followers: Optional[int] = Field(default=None, ge=0)
    max_followers: Optional[int] = Field(default=None, ge=0)
    min_engagement: Optional[float] = Field(default=None, ge=0.0)
    max_engagement: Optional[float] = Field(default=None, ge=0.0)

    location: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)


class CategorySearchRequest(BaseModel):
    category: str = Field(..., min_length=1)
    location: Optional[str] = Field(default=None)
    limit: int = Field(default=15, ge=1, le=200)

    min_followers: Optional[int] = Field(default=None, ge=0)
    max_followers: Optional[int] = Field(default=None, ge=0)
    min_engagement: Optional[float] = Field(default=None, ge=0.0)
    max_engagement: Optional[float] = Field(default=None, ge=0.0)


class EvaluationRequest(BaseModel):
    profiles: List[Dict[str, Any]] = Field(..., min_items=1, description="Profiles to evaluate")
    run_brightdata: bool = Field(default=False)
    run_llm: bool = Field(default=False)
    business_fit_query: Optional[str] = Field(default=None)
    max_profiles: Optional[int] = Field(default=None, ge=1, le=50000)
    max_posts: int = Field(default=6, ge=1, le=20)
    model: str = Field(default="gpt-5-mini")
    verbosity: str = Field(default="medium")
    concurrency: int = Field(default=64, ge=1, le=64)


class SearchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    count: int
    query: str
    method: str


class UsernameSearchResponse(BaseModel):
    success: bool
    result: Dict[str, Any]


class EvaluationResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    brightdata_results: List[Dict[str, Any]]
    profile_fit: List[Dict[str, Any]]
    count: int


class PipelineStageEvent(BaseModel):
    """Structured stage telemetry for the search pipeline."""

    stage: str
    data: Dict[str, Any]


class SearchPipelineRequest(BaseModel):
    """Run discovery plus optional enrichment in a single request."""

    search: SearchRequest
    run_brightdata: bool = Field(default=False)
    run_llm: bool = Field(default=False)
    business_fit_query: Optional[str] = Field(default=None)
    max_profiles: Optional[int] = Field(default=None, ge=1, le=50000)
    max_posts: int = Field(default=6, ge=1, le=20)
    model: str = Field(default="gpt-5-mini")
    verbosity: str = Field(default="medium")
    concurrency: int = Field(default=64, ge=1, le=64)


class SearchPipelineResponse(BaseModel):
    """Response returned by the staged search pipeline."""

    success: bool
    results: List[Dict[str, Any]]
    brightdata_results: List[Dict[str, Any]]
    profile_fit: List[Dict[str, Any]]
    stages: List[PipelineStageEvent]
    count: int


class BrightDataStageRequest(BaseModel):
    profiles: List[Dict[str, Any]] = Field(..., min_items=1)
    max_profiles: Optional[int] = Field(default=None, ge=1, le=50000)


class BrightDataStageResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    brightdata_results: List[Dict[str, Any]]
    count: int


class ProfileFitStageRequest(BaseModel):
    profiles: List[Dict[str, Any]] = Field(..., min_items=1)
    business_fit_query: str = Field(..., min_length=1)
    max_profiles: Optional[int] = Field(default=None, ge=1, le=50000)
    max_posts: int = Field(default=6, ge=1, le=20)
    model: str = Field(default="gpt-5-mini")
    verbosity: str = Field(default="medium")
    concurrency: int = Field(default=64, ge=1, le=64)
    use_brightdata: bool = Field(default=False)


class ProfileFitStageResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    brightdata_results: List[Dict[str, Any]]
    profile_fit: List[Dict[str, Any]]
    count: int


class ImageRefreshRequest(BaseModel):
    """Payload to refresh images for explicit usernames."""

    usernames: List[str] = Field(..., min_items=1, max_items=50)
    update_database: bool = Field(default=False)


class ImageRefreshSearchRequest(BaseModel):
    """Payload to refresh images for a batch of search results."""

    search_results: List[Dict[str, Any]] = Field(..., min_items=1)
    update_database: bool = Field(default=True)
