"""BrightData client helper for refreshing Instagram profile snapshots."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


@dataclass
class BrightDataConfig:
    api_key: str
    dataset_id: str
    poll_interval: int = 30


class BrightDataClient:
    """Wrapper around the BrightData dataset API."""

    BASE_URL = "https://api.brightdata.com/datasets/v3"

    def __init__(self, config: Optional[BrightDataConfig] = None) -> None:
        api_key = os.getenv("BRIGHTDATA_API_KEY") if config is None else config.api_key
        dataset_id = os.getenv("BRIGHTDATA_DATASET_ID") if config is None else config.dataset_id

        if not api_key or not dataset_id:
            raise RuntimeError(
                "BrightData configuration missing. Set BRIGHTDATA_API_KEY and BRIGHTDATA_DATASET_ID."
            )

        poll_interval = config.poll_interval if config else int(os.getenv("BRIGHTDATA_POLL_INTERVAL", "30"))

        self.config = BrightDataConfig(api_key=api_key, dataset_id=dataset_id, poll_interval=poll_interval)
        self.base_url = os.getenv("BRIGHTDATA_BASE_URL", self.BASE_URL)
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _trigger_job(self, url_objects: List[Dict[str, str]]) -> Optional[str]:
        response = requests.post(
            f"{self.base_url}/trigger",
            headers=self.headers,
            params={"dataset_id": self.config.dataset_id, "include_errors": "true"},
            json=url_objects,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("snapshot_id")

    def _wait_for_ready(self, snapshot_id: str) -> None:
        while True:
            response = requests.get(
                f"{self.base_url}/progress/{snapshot_id}",
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            status = payload.get("status")
            if status == "ready":
                return
            if status == "failed":
                raise RuntimeError(f"BrightData snapshot {snapshot_id} failed")
            time.sleep(self.config.poll_interval)

    def _download_csv(self, snapshot_id: str) -> pd.DataFrame:
        params = {"format": "csv"}
        response = requests.get(
            f"{self.base_url}/snapshot/{snapshot_id}",
            headers=self.headers,
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"brightdata_snapshot_{snapshot_id}_{timestamp}.csv"
        with open(filename, "wb") as file_handle:
            file_handle.write(response.content)
        return pd.read_csv(filename)

    def fetch_profiles(self, profile_urls: Iterable[str]) -> pd.DataFrame:
        """Trigger a BrightData job for the given profile URLs and return the result dataframe."""
        url_objects = [{"url": url} for url in profile_urls if url]
        if not url_objects:
            raise ValueError("No profile URLs provided to BrightData")

        snapshot_id = self._trigger_job(url_objects)
        if not snapshot_id:
            raise RuntimeError("Failed to trigger BrightData snapshot")

        self._wait_for_ready(snapshot_id)
        return self._download_csv(snapshot_id)

    @staticmethod
    def dataframe_to_profile_map(df: pd.DataFrame) -> Dict[str, Dict[str, Optional[str]]]:
        """Convert BrightData dataframe into a mapping keyed by profile URL or account."""
        profile_map: Dict[str, Dict[str, Optional[str]]] = {}
        for _, row in df.iterrows():
            profile_url = row.get("profile_url") or row.get("url")
            account = row.get("account")
            key = str(profile_url or account or "").strip().lower()
            if not key:
                continue
            profile_map[key] = row.to_dict()
        return profile_map
