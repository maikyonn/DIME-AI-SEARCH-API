"""Search API endpoints"""
import json
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from app.dependencies import get_search_engine
from app.models.search import (
    SearchRequest,
    SimilarSearchRequest,
    CategorySearchRequest,
    SearchResponse,
    ProfileFitTestRequest,
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


def format_engagement_rate(rate: float) -> float:
    """Convert engagement rate from decimal to percentage (multiply by 100) with full precision"""
    return rate * 100 if rate is not None else 0.0


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


def result_to_dict(result) -> Dict[str, Any]:
    """Convert SearchResult to dictionary for JSON serialization with original database column names"""
    return {
        'id': result.id,
        'account': result.account,
        'profile_name': result.profile_name,
        'followers': result.followers,
        'followers_formatted': format_number(result.followers),
        'avg_engagement': format_engagement_rate(result.avg_engagement),
        'avg_engagement_raw': result.avg_engagement,  # Keep raw value for advanced filtering
        'business_category_name': result.business_category_name,
        'business_address': result.business_address,
        'biography': result.biography,
        'profile_image_link': getattr(result, 'profile_image_link', ''),
        'profile_url': getattr(result, 'profile_url', ''),
        'business_email': getattr(result, 'business_email', ''),
        'email_address': getattr(result, 'email_address', ''),
        'posts': parse_json_field(getattr(result, 'posts', '')),
        'is_personal_creator': getattr(result, 'is_personal_creator', True),
        # Original database LLM score columns
        'individual_vs_org_score': getattr(result, 'individual_vs_org_score', 0),
        'generational_appeal_score': getattr(result, 'generational_appeal_score', 0),
        'professionalization_score': getattr(result, 'professionalization_score', 0),
        'relationship_status_score': getattr(result, 'relationship_status_score', 0),
        # Vector search similarity scores (text-based search)
        'keyword_score': getattr(result, 'keyword_score', 0.0),
        'profile_score': getattr(result, 'profile_score', 0.0),
        'content_score': getattr(result, 'content_score', 0.0),
        'combined_score': getattr(result, 'combined_score', 0.0),
        # Vector similarity scores (direct vector comparison)
        'keyword_similarity': getattr(result, 'keyword_similarity', 0.0),
        'profile_similarity': getattr(result, 'profile_similarity', 0.0),
        'content_similarity': getattr(result, 'content_similarity', 0.0),
        'vector_similarity_score': getattr(result, 'vector_similarity_score', 0.0),
        'similarity_explanation': getattr(result, 'similarity_explanation', ''),
        'fit_score': getattr(result, 'fit_score', None),
        'fit_rationale': getattr(result, 'fit_rationale', None),
        'fit_error': getattr(result, 'fit_error', None),
        'fit_prompt': getattr(result, 'fit_prompt', None),
        'fit_raw_response': getattr(result, 'fit_raw_response', None),
    }


@router.get("/username/{username}")
async def get_creator_by_username(
    username: str,
    search_engine=Depends(get_search_engine)
):
    """Retrieve a single creator profile by exact username."""
    sanitized = username.strip()
    if not sanitized:
        raise HTTPException(status_code=400, detail="Username is required")

    try:
        result = search_engine.get_creator_by_username(sanitized)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Creator '@{sanitized}' not found"
            )

        return {
            'success': True,
            'result': result_to_dict(result)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Username lookup failed: {str(e)}"
        )


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
        # Prepare custom weights if provided
        custom_weights = None
        if request.custom_weights:
            custom_weights = {
                'keyword': request.custom_weights.keyword,
                'profile': request.custom_weights.profile,
                'content': request.custom_weights.content
            }

        vector_query = request.vector_query or request.query
        business_query = request.business_query or request.business_fit_query

        # Use vector search with full customization
        results, debug_payload = search_engine.search_creators_for_campaign(
            query=vector_query,
            method=request.method,
            limit=request.limit,
            min_followers=request.min_followers,
            max_followers=request.max_followers,
            min_engagement=request.min_engagement,
            max_engagement=request.max_engagement,
            location_filter=request.location,
            target_category=request.category,
            relevance_keywords=request.keywords,
            custom_weights=custom_weights,
            similarity_threshold=request.similarity_threshold,
            return_vectors=request.return_vectors,
            business_fit_query=business_query,
            post_filter_limit=request.post_filter_limit,
            post_filter_concurrency=request.post_filter_concurrency or 8,
            post_filter_max_posts=request.post_filter_max_posts or 6,
            post_filter_model=request.post_filter_model or "gpt-5-mini",
            post_filter_verbosity=request.post_filter_verbosity or "medium",
            post_filter_use_brightdata=request.post_filter_use_brightdata or False,
            return_debug=True,
            # Account Type Filters
            is_verified=request.is_verified,
            is_business_account=request.is_business_account,
            # LLM Score Filters
            min_individual_vs_org_score=request.min_individual_vs_org_score,
            max_individual_vs_org_score=request.max_individual_vs_org_score,
            min_generational_appeal_score=request.min_generational_appeal_score,
            max_generational_appeal_score=request.max_generational_appeal_score,
            min_professionalization_score=request.min_professionalization_score,
            max_professionalization_score=request.max_professionalization_score,
            min_relationship_status_score=request.min_relationship_status_score,
            max_relationship_status_score=request.max_relationship_status_score,
            # Content Stats Filters
            min_posts_count=request.min_posts_count,
            max_posts_count=request.max_posts_count,
            min_following=request.min_following,
            max_following=request.max_following
        )
        
        # Convert results to dictionaries
        results_data = [result_to_dict(result) for result in results]
        
        debug_payload['vector_query'] = vector_query
        debug_payload['business_query'] = business_query

        return SearchResponse(
            success=True,
            results=results_data,
            count=len(results_data),
            query=vector_query,
            business_query=business_query,
            method=request.method,
            debug=debug_payload
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/profile-fit/test")
async def profile_fit_test(
    request: ProfileFitTestRequest,
    search_engine=Depends(get_search_engine)
):
    """Run stage-two profile fit scoring on a single influencer."""
    if not request.account and not request.profile_url:
        raise HTTPException(status_code=400, detail="Either account or profile_url must be provided")

    try:
        result = search_engine.run_profile_fit_preview(
            business_fit_query=request.business_fit_query,
            account=request.account,
            profile_url=request.profile_url,
            max_posts=request.max_posts,
            model=request.model,
            verbosity=request.verbosity,
            use_brightdata=request.use_brightdata,
            concurrency=request.concurrency,
        )

        return {
            'success': True,
            'result': {
                'account': result.account,
                'profile_url': result.profile_url,
                'followers': result.followers,
                'score': result.score,
                'rationale': result.rationale,
                'error': result.error,
                'prompt': result.prompt,
                'raw_response': result.raw_response,
            }
        }

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Profile fit test failed: {exc}") from exc


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
        # Prepare custom weights if provided
        custom_weights = None
        if request.custom_weights:
            custom_weights = {
                'keyword': request.custom_weights.keyword,
                'profile': request.custom_weights.profile,
                'content': request.custom_weights.content
            }
        
        results = search_engine.find_similar_creators(
            reference_account=request.account,
            limit=request.limit,
            min_followers=request.min_followers,
            max_followers=request.max_followers,
            min_engagement=request.min_engagement,
            max_engagement=request.max_engagement,
            location_filter=request.location,
            target_category=request.category,
            similarity_threshold=request.similarity_threshold,
            use_vector_similarity=request.use_vector_similarity,
            custom_weights=custom_weights
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
        custom_weights = None
        if request.custom_weights:
            custom_weights = {
                'keyword': request.custom_weights.keyword,
                'profile': request.custom_weights.profile,
                'content': request.custom_weights.content
            }

        results = search_engine.search_by_category(
            category=request.category,
            location=request.location,
            limit=request.limit,
            min_followers=request.min_followers,
            max_followers=request.max_followers,
            min_engagement=request.min_engagement,
            max_engagement=request.max_engagement,
            custom_weights=custom_weights,
            similarity_threshold=request.similarity_threshold
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
