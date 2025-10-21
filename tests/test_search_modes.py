import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402
from app.core.search_engine import SearchResult  # noqa: E402
from app.dependencies import get_search_engine  # noqa: E402


class StubSearchEngine:
    def __init__(self) -> None:
        self.calls = []
        self.evaluate_handler = None

    def search_creators_for_campaign(
        self,
        *,
        query: str,
        method: str,
        limit: int,
        **_: object,
    ):
        self.calls.append({"query": query, "method": method, "limit": limit})
        return [
            SearchResult(
                id=1,
                account="creator",
                profile_name="Creator",
                followers=1234,
                avg_engagement=0.05,
                business_category_name="lifestyle",
                business_address="",
                biography="Lifestyle creator in SF",
            )
        ]

    def evaluate_profiles(self, *args, **kwargs):  # pragma: no cover - unused here
        if self.evaluate_handler is not None:
            return self.evaluate_handler(*args, **kwargs)
        raise NotImplementedError


@pytest.fixture()
def api_client():
    stub = StubSearchEngine()
    app.dependency_overrides[get_search_engine] = lambda: stub
    client = TestClient(app)
    try:
        yield client, stub
    finally:
        app.dependency_overrides.pop(get_search_engine, None)


def test_lexical_search_accepts_high_limit(api_client):
    client, stub = api_client
    response = client.post(
        "/api/v1/search/",
        json={"query": "san francisco", "method": "lexical", "limit": 750},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert stub.calls[-1]["method"] == "lexical"
    assert stub.calls[-1]["limit"] == 750
    assert stub.calls[-1]["lexical_scope"] == 'bio'


def test_semantic_search_uses_vector_pipeline(api_client):
    client, stub = api_client
    response = client.post(
        "/api/v1/search/",
        json={"query": "gaming", "method": "semantic", "limit": 25},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert stub.calls[-1]["method"] == "semantic"


def test_hybrid_search_defaults(api_client):
    client, stub = api_client
    response = client.post(
        "/api/v1/search/",
        json={"query": "beauty tutorials", "limit": 30},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert stub.calls[-1]["method"] == "hybrid"
    assert stub.calls[-1]["limit"] == 30


def test_lexical_search_with_posts_scope(api_client):
    client, stub = api_client
    response = client.post(
        "/api/v1/search/",
        json={"query": "san francisco", "method": "lexical", "limit": 50, "lexical_scope": "bio_posts"},
    )
    assert response.status_code == 200
    assert stub.calls[-1]["lexical_scope"] == 'bio_posts'


def test_evaluation_stream_emits_progress_events(api_client):
    client, stub = api_client

    def evaluate_handler(profiles, *, progress_cb=None, **kwargs):
        if progress_cb:
            progress_cb(
                "evaluation_started",
                {"count": len(profiles), "run_brightdata": True, "run_llm": False},
            )
            progress_cb("brightdata_started", {"count": len(profiles)})
            progress_cb("brightdata_completed", {"count": len(profiles)})
        result = SearchResult(
            id=1,
            account="creator",
            profile_name="Creator",
            followers=1234,
            avg_engagement=0.05,
            business_category_name="lifestyle",
            business_address="",
            biography="Lifestyle creator in SF",
        )
        return [result], {"brightdata_results": [], "profile_fit": []}

    stub.evaluate_handler = evaluate_handler

    stages = []
    with client.stream(
        "POST",
        "/api/v1/search/evaluate/stream",
        json={
            "profiles": [{"account": "creator"}],
            "run_brightdata": True,
            "run_llm": False,
        },
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if isinstance(line, bytes):
                line = line.decode()
            if not line:
                continue
            if line.startswith("data:"):
                payload = json.loads(line.split(":", 1)[1].strip())
                stages.append(payload.get("stage"))
                if payload.get("stage") == "completed":
                    break

    assert "evaluation_started" in stages
    assert "brightdata_started" in stages
    assert stages[-1] == "completed"
