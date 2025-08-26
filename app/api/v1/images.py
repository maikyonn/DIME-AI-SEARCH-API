"""Image refresh API endpoints"""
import asyncio
import os
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse
import requests
from urllib.parse import urlparse

from app.dependencies import get_image_refresh_service, get_optional_search_engine
from app.models.search import ImageRefreshRequest, ImageRefreshSearchRequest
from app.models.creator import ImageRefreshResult, ImageRefreshSummary


router = APIRouter()


@router.post("/refresh")
async def refresh_images(
    request: ImageRefreshRequest,
    image_service=Depends(get_image_refresh_service)
):
    """
    Refresh images for specified users
    
    This endpoint fetches fresh profile images and data from Bright Data
    for the specified usernames.
    """
    try:
        # Call the async refresh method
        results = await image_service.refresh_images_for_users(request.usernames)
        
        # Prepare response
        response_data = {
            'success': True,
            'results': [
                ImageRefreshResult(
                    username=result.username,
                    success=result.success,
                    profile_image_url=result.profile_image_url,
                    error=result.error if not result.success else None
                ).dict()
                for result in results
            ],
            'summary': ImageRefreshSummary(
                total=len(results),
                successful=sum(1 for r in results if r.success),
                failed=sum(1 for r in results if not r.success)
            ).dict()
        }
        
        # Note: Database update functionality disabled in original service
        if request.update_database:
            response_data['database_update'] = {
                'status': 'disabled - schema incompatible',
                'message': 'Database updates are currently disabled due to schema limitations'
            }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image refresh failed: {str(e)}"
        )


@router.post("/refresh/search-results")
async def refresh_images_for_search_results(
    request: ImageRefreshSearchRequest,
    image_service=Depends(get_image_refresh_service)
):
    """
    Refresh images for users returned by a search
    
    This endpoint extracts usernames from search results and refreshes
    their profile images and data.
    """
    try:
        # Extract usernames from search results
        usernames = []
        for result in request.search_results:
            if isinstance(result, dict) and 'account' in result:
                usernames.append(result['account'])
        
        if not usernames:
            raise HTTPException(
                status_code=400,
                detail="No valid usernames found in search results"
            )
        
        # Limit to reasonable number
        if len(usernames) > 50:
            usernames = usernames[:50]
        
        # Call the async refresh method
        results = await image_service.refresh_images_for_users(usernames)
        
        # Prepare response
        response_data = {
            'success': True,
            'refreshed_usernames': usernames,
            'results': [
                ImageRefreshResult(
                    username=result.username,
                    success=result.success,
                    profile_image_url=result.profile_image_url,
                    error=result.error if not result.success else None
                ).dict()
                for result in results
            ],
            'summary': ImageRefreshSummary(
                total=len(results),
                successful=sum(1 for r in results if r.success),
                failed=sum(1 for r in results if not r.success)
            ).dict()
        }
        
        # Note: Database update functionality disabled in original service
        if request.update_database:
            response_data['database_update'] = {
                'status': 'disabled - schema incompatible',
                'message': 'Database updates are currently disabled due to schema limitations'
            }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image refresh for search results failed: {str(e)}"
        )


@router.get("/refresh/job/{snapshot_id}")
async def get_refresh_job_status(
    snapshot_id: str,
    image_service=Depends(get_image_refresh_service)
):
    """
    Get status of a running image refresh job
    
    This endpoint allows you to check the status of a previously initiated
    image refresh job using its snapshot ID.
    """
    try:
        job_status = image_service.get_job_status(snapshot_id)
        
        if job_status:
            return {
                'success': True,
                'job': job_status
            }
        else:
            raise HTTPException(
                status_code=404,
                detail='Job not found'
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )


@router.get("/refresh/status")
async def get_service_status():
    """
    Get status of the image refresh service
    
    This endpoint provides information about the availability and current
    state of the image refresh service.
    """
    from app.dependencies import _image_refresh_service
    
    api_token_available = bool(os.getenv("BRIGHTDATA_API_TOKEN"))
    service_available = _image_refresh_service is not None
    
    return {
        'service_available': service_available,
        'api_token_configured': api_token_available,
        'active_jobs': _image_refresh_service.active_jobs_count if _image_refresh_service else 0
    }


@router.get("/proxy")
async def proxy_image(url: str = Query(..., description="Image URL to proxy")):
    """
    Proxy Instagram images to bypass CORS restrictions
    
    This endpoint acts as a proxy to fetch Instagram images and serve them
    with appropriate CORS headers to bypass browser restrictions.
    """
    if not url:
        raise HTTPException(status_code=400, detail='No URL provided')
    
    # Validate that URL is from Instagram domains
    parsed_url = urlparse(url)
    hostname = parsed_url.netloc.lower()
    
    # Instagram uses various CDN domains - allow all legitimate patterns
    allowed_patterns = [
        'cdninstagram.com',  # scontent-*.cdninstagram.com
        'fna.fbcdn.net',     # instagram.*.fna.fbcdn.net
        'instagram.com'      # direct instagram.com domains
    ]
    
    # Check if domain matches any Instagram CDN pattern
    is_valid_domain = any(
        hostname.endswith(pattern) or 
        (pattern == 'cdninstagram.com' and 'scontent-' in hostname and hostname.endswith('cdninstagram.com')) or
        (pattern == 'fna.fbcdn.net' and 'instagram.' in hostname and hostname.endswith('fna.fbcdn.net'))
        for pattern in allowed_patterns
    )
    
    if not is_valid_domain:
        raise HTTPException(status_code=403, detail='Invalid domain')
    
    try:
        # Fetch the image from Instagram
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.instagram.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }
        
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        
        if response.status_code == 200:
            # Get content type from the response
            content_type = response.headers.get('content-type', 'image/jpeg')
            
            # Create a streaming response
            def generate():
                for chunk in response.iter_content(chunk_size=8192):
                    yield chunk
            
            return StreamingResponse(
                generate(),
                media_type=content_type,
                headers={
                    'Cache-Control': 'public, max-age=3600',  # Cache for 1 hour
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail='Failed to fetch image'
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail='Request timeout')
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f'Request failed: {str(e)}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Proxy error: {str(e)}')