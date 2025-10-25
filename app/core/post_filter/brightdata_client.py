"""BrightData client that proxies through the DIME-AI-BD service."""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

from app.config import settings


class BrightDataClient:
    """Wrapper around the DIME-AI-BD HTTP API for BrightData snapshots."""

    def __init__(self) -> None:
        base_url = (settings.BRIGHTDATA_SERVICE_URL or "").rstrip("/")
        if not base_url:
            raise RuntimeError("BRIGHTDATA_SERVICE_URL must be configured to use BrightData features")

        self.base_url = base_url
        self.poll_interval = max(1, settings.BRIGHTDATA_JOB_POLL_INTERVAL or 5)
        self.job_timeout = settings.BRIGHTDATA_JOB_TIMEOUT or 600
        self.session = requests.Session()

    def fetch_profiles(self, profile_urls: Iterable[str]) -> pd.DataFrame:
        usernames = self._extract_usernames(profile_urls)
        if not usernames:
            raise ValueError("No profile URLs provided to BrightData service")

        payload = {"usernames": usernames, "update_database": False}
        response = self.session.post(f"{self.base_url}/refresh", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        job_id = data.get("job_id")
        if not job_id:
            result = data.get("result")
            if result:
                return self._records_to_dataframe(result.get("records", []))
            raise RuntimeError("BrightData service did not return job_id")

        result_payload = self._wait_for_job(job_id)
        records = result_payload.get("records", [])
        return self._records_to_dataframe(records)

    def _wait_for_job(self, job_id: str) -> Dict[str, object]:
        deadline = time.monotonic() + self.job_timeout
        last_status: Optional[str] = None
        while time.monotonic() < deadline:
            response = self.session.get(f"{self.base_url}/refresh/job/{job_id}", timeout=30)
            if response.status_code == 404:
                raise RuntimeError(f"BrightData job '{job_id}' not found")
            response.raise_for_status()
            payload = response.json().get("job", {})
            status = payload.get("status")
            last_status = status or last_status
            if status == "finished":
                return payload.get("result", {})
            if status == "failed":
                raise RuntimeError(payload.get("error") or f"BrightData job '{job_id}' failed")
            time.sleep(self.poll_interval)

        raise RuntimeError(
            f"Timed out after {self.job_timeout}s waiting for BrightData job '{job_id}' (last status: {last_status})"
        )

    def _extract_usernames(self, profile_urls: Iterable[str]) -> List[str]:
        usernames: List[str] = []
        for raw in profile_urls:
            platform, handle = self._parse_social_url(raw)
            if not handle:
                continue
            if platform == "instagram":
                usernames.append(handle)
        return usernames

    @staticmethod
    def _parse_social_url(url: str) -> Tuple[str, Optional[str]]:
        if not url:
            return "", None

        try:
            parsed = urlparse(url)
        except Exception:
            return "", None

        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").strip("/")
        if "instagram.com" in host and path:
            handle = path.split("/")[0].lstrip("@")
            return "instagram", handle
        if "tiktok.com" in host and path:
            if path.startswith("@"):
                handle = path.lstrip("@")
            elif path.startswith("/@"):
                handle = path[2:]
            else:
                handle = path
            return "tiktok", handle
        return "", None

    @staticmethod
    def _records_to_dataframe(records: List[Dict[str, object]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        return pd.DataFrame.from_records(records)

    @staticmethod
    def dataframe_to_profile_map(df: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
        profile_map: Dict[str, Dict[str, Optional[str]]] = {}
        if df.empty:
            return profile_map
        for _, row in df.iterrows():
            profile_url = row.get("profile_url") or row.get("url")
            account = row.get("account")
            key = str(profile_url or account or "").strip().lower()
            if not key:
                continue
            profile_map[key] = row.to_dict()
        return profile_map
