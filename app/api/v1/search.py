"""Search API endpoints"""
import json
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from app.dependencies import get_search_engine
from app.models.search import (
    SearchRequest, SimilarSearchRequest, CategorySearchRequest, SearchResponse
)
from app.models.creator import SearchResult


router = APIRouter()


def format_number(num: int) -> str:
    """Format numbers for display (e.g., 1.2K, 1.5M)"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(int(num))


def parse_json_field(json_str: str) -> List[Any]:
    """Parse JSON fields and extract meaningful content"""
    if not json_str or json_str == '':
        return []
    
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data[:3]  # Show first 3 items
        elif isinstance(data, dict):
            return [data]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def _is_business_description(query: str) -> bool:
    """Detect if a query is a business description rather than a search term"""
    query_lower = query.lower()
    
    # Business description indicators
    business_indicators = [
        'we are', 'we\'re', 'our company', 'our business', 'my business', 'my company',
        'i\'m', 'i am', 'startup', 'company', 'business', 'brand', 'we sell', 'we offer',
        'our mission', 'our goal', 'we want to', 'we need', 'looking for', 'targeting'
    ]
    
    # Check for business language
    for indicator in business_indicators:
        if indicator in query_lower:
            return True
    
    # Check for sentence structure (longer descriptions with multiple clauses)
    if len(query.split()) > 10 and any(word in query_lower for word in ['and', 'for', 'to', 'with']):
        return True
    
    return False


def result_to_dict(result) -> Dict[str, Any]:
    """Convert SearchResult to dictionary for JSON serialization"""
    return {
        'id': result.id,
        'account': result.account,
        'profile_name': result.profile_name,
        'followers': result.followers,
        'followers_formatted': format_number(result.followers),
        'avg_engagement': result.avg_engagement,
        'business_category_name': result.business_category_name,
        'business_address': result.business_address,
        'biography': result.biography,
        'profile_image_link': getattr(result, 'profile_image_link', ''),
        'posts': parse_json_field(getattr(result, 'posts', '')),
        'score': result.score,
        'engagement_score': result.engagement_score,
        'relevance_score': getattr(result, 'category_relevance_score', 0.0),
        'genz_appeal_score': result.genz_appeal_score,
        'authenticity_score': result.authenticity_score,
        'campaign_value_score': result.campaign_value_score,
        'category_relevance_score': getattr(result, 'category_relevance_score', 0.0),
        'business_alignment_score': getattr(result, 'business_alignment_score', 0.0),
        'is_personal_creator': getattr(result, 'is_personal_creator', True),
    }


@router.post("/", response_model=SearchResponse)
async def search_creators(
    request: SearchRequest,
    search_engine=Depends(get_search_engine)
):
    """
    Search for creators/influencers based on business description or query
    
    This endpoint supports both business description matching and general search.
    It automatically detects the query type or you can specify it explicitly.
    """
    try:
        # Convert custom weights if provided
        custom_weights = None
        if request.weights:
            custom_weights = {
                'business_alignment': request.weights.business_alignment,
                'genz_appeal': request.weights.genz_appeal,
                'authenticity': request.weights.authenticity,
                'engagement': request.weights.engagement,
                'campaign_value': request.weights.campaign_value
            }
        
        # Determine if this is a business description or regular search
        is_business_desc = request.is_business_description or _is_business_description(request.query)
        
        if is_business_desc:
            # Use business matching
            results = search_engine.match_creators_to_business(
                business_description=request.query,
                method=request.method,
                limit=request.limit,
                custom_weights=custom_weights,
                min_followers=request.min_followers,
                max_followers=request.max_followers,
                min_engagement=request.min_engagement,
                location_filter=request.location,
                target_category=request.category
            )
        else:
            # Use regular search
            results = search_engine.search_creators_for_campaign(
                query=request.query,
                method=request.method,
                limit=request.limit,
                min_followers=request.min_followers,
                max_followers=request.max_followers,
                min_engagement=request.min_engagement,
                location_filter=request.location,
                target_category=request.category,
                relevance_keywords=request.keywords,
                custom_weights=custom_weights
            )
        
        # Convert results to dictionaries
        results_data = [result_to_dict(result) for result in results]
        
        return SearchResponse(
            success=True,
            results=results_data,
            count=len(results_data),
            query=request.query,
            method=request.method
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/similar", response_model=SearchResponse)
async def find_similar_creators(
    request: SimilarSearchRequest,
    search_engine=Depends(get_search_engine)
):
    """
    Find creators similar to a reference account
    
    This endpoint finds creators with similar characteristics, engagement patterns,
    and audience demographics to the specified reference account.
    """
    try:
        results = search_engine.find_similar_creators(
            reference_account=request.account,
            limit=request.limit,
            min_followers=request.min_followers
        )
        
        results_data = [result_to_dict(result) for result in results]
        
        return SearchResponse(
            success=True,
            results=results_data,
            count=len(results_data),
            query=f"Similar to @{request.account}",
            method="similarity"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Similar search failed: {str(e)}"
        )


@router.post("/category", response_model=SearchResponse)
async def search_by_category(
    request: CategorySearchRequest,
    search_engine=Depends(get_search_engine)
):
    """
    Search creators by business category
    
    This endpoint searches for creators within a specific business category,
    optionally filtered by location.
    """
    try:
        results = search_engine.search_by_category(
            category=request.category,
            location=request.location,
            min_followers=request.min_followers,
            limit=request.limit
        )
        
        results_data = [result_to_dict(result) for result in results]
        
        return SearchResponse(
            success=True,
            results=results_data,
            count=len(results_data),
            query=f"Category: {request.category}" + (f" in {request.location}" if request.location else ""),
            method="category"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Category search failed: {str(e)}"
        )