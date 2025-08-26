"""Main API router for v1 endpoints"""
from fastapi import APIRouter

from app.api.v1 import search, creators, images

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(search.router, prefix="/search", tags=["Search"])
api_router.include_router(creators.router, prefix="/creators", tags=["Creators"])
api_router.include_router(images.router, prefix="/images", tags=["Images"])