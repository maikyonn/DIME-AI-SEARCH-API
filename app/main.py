from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager

from app.api.v1.router import api_router
from app.dependencies import get_search_engine, init_search_engine, init_image_refresh_service
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events"""
    # Startup
    print("🚀 Starting Gen Z Creator Search FastAPI...")
    
    # Initialize search engine
    if not init_search_engine():
        print("⚠️  Database not found. You'll need to upload data first.")
    else:
        print("✅ Search engine initialized successfully")
    
    # Initialize image refresh service
    init_image_refresh_service()
    
    print("📡 FastAPI server ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down FastAPI server...")


# Create FastAPI app
app = FastAPI(
    title="GenZ Creator Search API",
    description="API for searching and managing GenZ creators/influencers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "GenZ Creator Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    search_engine = get_search_engine()
    
    return {
        "status": "healthy",
        "database_available": search_engine is not None,
        "timestamp": "2024-01-01T00:00:00Z"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    print(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )