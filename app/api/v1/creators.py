"""Creator-related API endpoints"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.dependencies import get_search_engine
from app.models.creator import CreatorSummary


router = APIRouter()


@router.get("/{creator_id}")
async def get_creator_detail(
    creator_id: int,
    search_engine=Depends(get_search_engine)
):
    """
    Get detailed information about a specific creator
    
    Note: This is a placeholder endpoint. Full implementation would require
    additional database queries to get comprehensive creator details.
    """
    # This would require additional implementation in the original search engine
    # For now, return a basic response
    raise HTTPException(
        status_code=501,
        detail="Creator detail endpoint not yet implemented. Use search endpoints instead."
    )


@router.get("/")
async def list_creators(
    limit: int = 20,
    offset: int = 0,
    category: str = None,
    location: str = None,
    search_engine=Depends(get_search_engine)
):
    """
    List creators with optional filtering
    
    This endpoint would provide general creator listing functionality.
    Currently redirects to search endpoints for better filtering options.
    """
    raise HTTPException(
        status_code=501,
        detail="General creator listing not implemented. Use /search endpoints with specific criteria."
    )