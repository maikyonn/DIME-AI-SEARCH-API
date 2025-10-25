"""HTTP client that proxies image refresh requests to the BrightData service."""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

import requests

from app.config import settings
from app.models.creator import ImageRefreshResult


class BrightDataAPIError(RuntimeError):
    """Raised when the downstream BrightData API returns an error."""


class FastAPIImageRefreshService:
    """Calls the DIME-AI-BD service to refresh profile images via BrightData."""

    def __init__(self) -> None:
        base_url = (settings.BRIGHTDATA_SERVICE_URL or "").rstrip("/")
        self.base_url = base_url or None
        self.poll_interval = max(1, settings.BRIGHTDATA_JOB_POLL_INTERVAL or 5)
        self.job_timeout = settings.BRIGHTDATA_JOB_TIMEOUT or 600
        self.session = requests.Session()

    @property
    def is_available(self) -> bool:
        return bool(self.base_url)

    async def refresh_images_for_users(self, usernames: List[str]) -> List[ImageRefreshResult]:
        if not self.is_available:
            raise BrightDataAPIError("BrightData service URL is not configured")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._refresh_blocking, usernames)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, object]]:
        if not self.is_available:
            return None
        response = self.session.get(f"{self.base_url}/refresh/job/{job_id}", timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload.get("job")

    @property
    def active_jobs_count(self) -> int:
        if not self.is_available:
            return 0
        try:
            response = self.session.get(f"{self.base_url}/refresh/status", timeout=15)
            response.raise_for_status()
            data = response.json()
            return int(data.get("active_jobs", 0))
        except Exception:  # pragma: no cover - status calls are best effort
            return 0

    def _refresh_blocking(self, usernames: List[str]) -> List[ImageRefreshResult]:
        payload = {
            "usernames": usernames,
            "update_database": False,
        }
        response = self.session.post(f"{self.base_url}/refresh", json=payload, timeout=30)
        response.raise_for_status()
        job_id = response.json().get("job_id")
        if not job_id:
            raise BrightDataAPIError("BrightData service did not return a job_id")
        return self._wait_for_job(job_id)

    def _wait_for_job(self, job_id: str) -> List[ImageRefreshResult]:
        deadline = time.monotonic() + self.job_timeout
        last_status: Optional[str] = None
        while time.monotonic() < deadline:
            job_payload = self.get_job_status(job_id)
            if not job_payload:
                raise BrightDataAPIError(f"BrightData job '{job_id}' not found")

            status = job_payload.get("status")
            last_status = status or last_status
            if status == "finished":
                result = job_payload.get("result") or {}
                raw_results = result.get("results", [])
                return [ImageRefreshResult(**item) for item in raw_results]
            if status == "failed":
                error = job_payload.get("error")
                raise BrightDataAPIError(error or f"BrightData job '{job_id}' failed")

            time.sleep(self.poll_interval)

        raise BrightDataAPIError(
            f"Timed out after {self.job_timeout}s waiting for BrightData job '{job_id}' (last status: {last_status})"
        )
