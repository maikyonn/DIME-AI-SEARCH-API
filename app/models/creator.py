"""Creator-related Pydantic models"""
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Creator search result model"""
    id: int
    account: str
    profile_name: str
    followers: int
    followers_formatted: str
    avg_engagement: float
    business_category_name: str
    business_address: str
    biography: str
    profile_image_link: Optional[str] = ""
    posts: List[Any] = []
    score: float = 0.0
    engagement_score: float = 0.0
    relevance_score: float = 0.0
    genz_appeal_score: float = 0.0
    authenticity_score: float = 0.0
    campaign_value_score: float = 0.0
    category_relevance_score: float = 0.0
    business_alignment_score: float = 0.0
    is_personal_creator: bool = True


class CreatorSummary(BaseModel):
    """Basic creator information"""
    account: str
    profile_name: str
    followers: int
    followers_formatted: str
    profile_image_link: Optional[str] = ""


class ImageRefreshResult(BaseModel):
    """Result of image refresh operation"""
    username: str
    success: bool
    profile_image_url: Optional[str] = None
    error: Optional[str] = None


class ImageRefreshSummary(BaseModel):
    """Summary of image refresh operation"""
    total: int
    successful: int
    failed: int