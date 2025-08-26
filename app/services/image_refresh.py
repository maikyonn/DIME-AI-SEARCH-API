"""
FastAPI wrapper for image refresh service.
This module provides a clean interface to the existing image refresh functionality.
"""
import os
import sys
from typing import List, Optional

# Add the original src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(os.path.join(project_root, "src"))

# Import from the original services
from services.image_refresh_service import create_image_refresh_service, ImageRefreshResult


class FastAPIImageRefreshService:
    """FastAPI wrapper for the image refresh service"""
    
    def __init__(self):
        self.service = create_image_refresh_service()
    
    @property
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self.service is not None
    
    async def refresh_images_for_users(self, usernames: List[str]) -> List[ImageRefreshResult]:
        """Refresh images for specified users"""
        if not self.service:
            raise RuntimeError("Image refresh service not available")
        
        return await self.service.refresh_images_for_users(usernames)
    
    def get_job_status(self, snapshot_id: str):
        """Get status of a running job"""
        if not self.service:
            return None
        
        return self.service.get_job_status(snapshot_id)
    
    @property
    def active_jobs_count(self) -> int:
        """Get number of active jobs"""
        if not self.service:
            return 0
        
        return len(self.service.active_jobs)