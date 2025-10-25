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
        self.profile_fit_payload = {
            "brightdata_results": [],
            "profile_fit": [
                {
                    "account": "creator",
                    "score": 7,
                    "rationale": "Great fit",
                }
            ],
        }

    def search_creators_for_campaign(
        self,
        *,
        query: str,
        method: str,
        limit: int,
        **kwargs: object,
    ):
        call = {"query": query, "method": method, "limit": limit}
        call.update({k: v for k, v in kwargs.items()})
        self.calls.append(call)
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

    def run_brightdata_stage(self, profiles, *, max_profiles=None, progress_cb=None):
        result = SearchResult(
            id=3,
            account="creator",
            profile_name="Creator",
            followers=5555,
            avg_engagement=0.02,
            business_category_name="travel",
            business_address="",
            biography="Travel creator",
        )
        return [result], {"brightdata_results": [{"account": "creator"}]}

    def run_profile_fit_stage(
        self,
        profiles,
        *,
        business_fit_query: str,
        max_profiles=None,
        concurrency=64,
        max_posts=6,
        model="gpt-5-mini",
        verbosity="medium",
        use_brightdata=False,
        progress_cb=None,
    ):
        result = SearchResult(
            id=4,
            account="creator",
            profile_name="Creator",
            followers=8888,
            avg_engagement=0.09,
            business_category_name="tech",
            business_address="",
            biography="Tech creator",
        )
        return [result], self.profile_fit_payload


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
    response = client.post("/search/", json={"query": "san francisco", "method": "lexical", "limit": 750})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert stub.calls[-1]["method"] == "lexical"
    assert stub.calls[-1]["limit"] == 750
    assert stub.calls[-1]["lexical_scope"] == 'bio'


def test_semantic_search_uses_vector_pipeline(api_client):
    client, stub = api_client
    response = client.post("/search/", json={"query": "gaming", "method": "semantic", "limit": 25})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert stub.calls[-1]["method"] == "semantic"


def test_hybrid_search_defaults(api_client):
    client, stub = api_client
    response = client.post("/search/", json={"query": "beauty tutorials", "limit": 30})
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert stub.calls[-1]["method"] == "hybrid"
    assert stub.calls[-1]["limit"] == 30


def test_lexical_search_with_posts_scope(api_client):
    client, stub = api_client
    response = client.post(
        "/search/", json={"query": "san francisco", "method": "lexical", "limit": 50, "lexical_scope": "bio_posts"}
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
        "/search/evaluate/stream",
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


def test_pipeline_endpoint_returns_stage_history(api_client):
    client, _ = api_client
    response = client.post(
        "/search/pipeline",
        json={
            "search": {
                "query": "creator",
                "method": "hybrid",
                "limit": 5,
            },
            "run_brightdata": False,
            "run_llm": False,
            "max_profiles": 1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    stage_names = [stage["stage"] for stage in payload["stages"]]
    assert stage_names[:2] == ["search_started", "search_completed"]
    assert stage_names[-2:] == ["evaluation_skipped", "completed"]
    assert payload["count"] == 1
    assert payload["results"][0]["account"] == "creator"


def test_pipeline_stream_emits_search_and_evaluation_events(api_client):
    client, stub = api_client

    def evaluate_handler(profiles, *, progress_cb=None, **kwargs):
        assert len(profiles) == 1
        if progress_cb:
            progress_cb(
                "evaluation_started",
                {"count": len(profiles), "run_brightdata": True, "run_llm": False},
            )
            progress_cb("brightdata_started", {"count": len(profiles)})
            progress_cb("brightdata_completed", {"count": len(profiles)})
        result = SearchResult(
            id=2,
            account="creator",
            profile_name="Creator",
            followers=9876,
            avg_engagement=0.07,
            business_category_name="fashion",
            business_address="",
            biography="Fashion creator in NY",
        )
        return [result], {"brightdata_results": [], "profile_fit": []}

    stub.evaluate_handler = evaluate_handler

    stages = []
    with client.stream(
        "POST",
        "/search/pipeline/stream",
        json={
            "search": {
                "query": "creator",
                "method": "hybrid",
                "limit": 1,
            },
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

    assert "search_started" in stages
    assert "evaluation_started" in stages
    assert stages[-1] == "completed"


def test_brightdata_stage_endpoint_returns_records(api_client):
    client, _ = api_client
    response = client.post(
        "/search/evaluate/brightdata",
        json={"profiles": [{"account": "creator"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["account"] == "creator"
    assert payload["brightdata_results"]


def test_llm_stage_endpoint_scores_profiles(api_client):
    client, stub = api_client
    response = client.post(
        "/search/evaluate/llm",
        json={
            "profiles": [{"account": "creator"}],
            "business_fit_query": "looking for creators",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["account"] == "creator"
    assert payload["profile_fit"] == stub.profile_fit_payload["profile_fit"]
