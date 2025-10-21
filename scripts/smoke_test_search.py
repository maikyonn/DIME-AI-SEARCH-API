#!/usr/bin/env python3
"""Quick smoke test for the influencer_facets search backend."""
from __future__ import annotations

import json
from typing import Any

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.core.search_engine import FastAPISearchEngine
from app.config import settings


def dump_result(result: Any) -> dict[str, Any]:
    return {
        "account": getattr(result, "account", None),
        "followers": getattr(result, "followers", None),
        "platform": getattr(result, "platform", None),
        "combined_score": getattr(result, "combined_score", None),
    }


def main() -> None:
    engine = FastAPISearchEngine(settings.DB_PATH)
    results, _debug = engine.search_creators_for_campaign(
        query="fitness influencers",
        method="hybrid",
        limit=5,
        min_followers=1000,
        return_debug=True,
    )
    payload = [dump_result(item) for item in results]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
