"""
FastAPI wrapper for image refresh service.
This module provides a clean interface to the existing image refresh functionality.
"""
import os
import sys
from typing import List, Optional, Any

# Add the original src directory to path (if available)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
dime_db_root = os.path.join(project_root, "DIME-AI-DB")
dime_db_src = os.path.join(dime_db_root, "src")

for path in (dime_db_root, dime_db_src):
    if path not in sys.path and os.path.isdir(path):
        sys.path.append(path)

# Import from the original services (optional dependency)
try:
    from services.image_refresh_service import create_image_refresh_service, ImageRefreshResult  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    create_image_refresh_service = None
    ImageRefreshResult = Any


class FastAPIImageRefreshService:
    """FastAPI wrapper for the image refresh service"""
    
    def __init__(self):
        if callable(create_image_refresh_service):
            self.service = create_image_refresh_service()
        else:
            self.service = None
    
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
