"""Shared dependencies for FastAPI endpoints"""
import os
from typing import Optional

from fastapi import Depends, HTTPException

from app.config import settings

# Global instances
_search_engine = None
_text_search_engine = None
_image_refresh_service = None
_post_filter_ready = False


def init_search_engine() -> bool:
    """Initialize the search engine"""
    global _search_engine, _post_filter_ready
    try:
        from app.core.search_engine import FastAPISearchEngine

        db_path = settings.DB_PATH
        if not db_path:
            print("❌ DB_PATH is not configured; set DB_PATH or ensure default resolution succeeds.")
            return False

        if not os.path.exists(db_path):
            print(f"❌ LanceDB database not found at: {db_path}")
            print("   Ensure DIME-AI-DB/data/lancedb is available or set DB_PATH to the correct location.")
            return False

        _search_engine = FastAPISearchEngine(db_path)
        print("✅ Search engine initialized")
        print(f"   • DB path: {db_path}")
        _post_filter_ready = True
        return True
    except Exception as e:
        print(f"Error initializing search engine: {e}")
        return False


def init_post_filter() -> None:
    from app.config import settings
    if settings.OPENAI_API_KEY and settings.BRIGHTDATA_SERVICE_URL:
        print("✅ Post-filter pipeline ready (LLM + BrightData service)")
    else:
        print("⚠️ Post-filter pipeline missing OPENAI_API_KEY or BRIGHTDATA_SERVICE_URL; stage two will be limited")


def init_image_refresh_service() -> bool:
    """Initialize the image refresh service"""
    global _image_refresh_service
    try:
        from app.services.image_refresh import FastAPIImageRefreshService

        _image_refresh_service = FastAPIImageRefreshService()
        if _image_refresh_service.is_available:
            print("✅ Image refresh service initialized successfully")
            return True
        else:
            print("⚠️ Image refresh service could not be initialized (missing API token)")
            return False
    except Exception as e:
        print(f"❌ Error initializing image refresh service: {e}")
        return False


def get_search_engine():
    """Dependency to get search engine instance"""
    if _search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Please ensure database is available."
        )
    return _search_engine


def get_image_refresh_service():
    """Dependency to get image refresh service instance"""
    if _image_refresh_service is None or not _image_refresh_service.is_available:
        raise HTTPException(
            status_code=503,
            detail="Image refresh service not available. Configure BRIGHTDATA_SERVICE_URL to point at DIME-AI-BD.",
        )
    return _image_refresh_service


async def get_optional_search_engine():
    """Get search engine if available, None otherwise"""
    return _search_engine


async def get_optional_image_refresh_service():
    """Get image refresh service if available, None otherwise"""
    return _image_refresh_service


def init_text_search_engine() -> bool:
    """Initialize the plaintext biography search engine."""
    global _text_search_engine
    try:
        from app.core.text_search import TextSearchEngine

        table_path = settings.TEXT_DB_PATH or settings.DB_PATH
        if not table_path:
            print("⚠️ TEXT_DB_PATH is not configured and DB_PATH is unavailable.")
            return False

        if not os.path.exists(table_path):
            print(f"⚠️ Biography dataset not found at: {table_path}")
            print("   Expected LanceDB directory inside DIME-AI-DB/data/lancedb.")
            return False

        if _search_engine is None and not init_search_engine():
            print("⚠️ Unable to initialize primary search engine; text search unavailable")
            return False

        _text_search_engine = TextSearchEngine(
            table_path=table_path,
            table_name=settings.TABLE_NAME or "influencer_facets",
            vector_engine=_search_engine,
        )
        print("✅ Text search engine initialized")
        print(f"   • Text DB path: {table_path}")
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error initializing text search engine: {exc}")
        _text_search_engine = None
        return False


def get_text_search_engine():
    """Dependency to get text search engine instance"""
    if _text_search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Text search engine not initialized."
        )
    return _text_search_engine


async def get_optional_text_search_engine():
    """Get text search engine if available, None otherwise"""
    return _text_search_engine
