"""Simplified Search API endpoints."""
import json
import logging
import threading
from queue import SimpleQueue
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.dependencies import get_search_engine
from app.models.search import (
    SearchRequest,
    SimilarSearchRequest,
    CategorySearchRequest,
    EvaluationRequest,
    SearchResponse,
    EvaluationResponse,
    UsernameSearchResponse,
)

router = APIRouter()

logger = logging.getLogger("search_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[SearchAPI] %(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _format_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def _format_engagement(rate: float) -> float:
    return rate * 100 if rate is not None else 0.0


def _parse_posts(raw_posts: Any) -> List[Any]:
    if not raw_posts:
        return []
    if isinstance(raw_posts, list):
        return raw_posts[:3]
    if isinstance(raw_posts, dict):
        return [raw_posts]
    return []


def result_to_dict(result) -> Dict[str, Any]:
    platform = getattr(result, "platform", None)
    platform_normalized = platform.lower() if isinstance(platform, str) else None
    profile_image = getattr(result, "profile_image_link", "") or getattr(
        result, "profile_image_url", ""
    )

    account_value = getattr(result, "account", "")
    profile_url = getattr(result, "profile_url", "") or getattr(result, "url", "")
    if not profile_url and account_value:
        if platform_normalized == "tiktok":
            profile_url = f"https://www.tiktok.com/@{account_value}"
        else:
            profile_url = f"https://instagram.com/{account_value}"

    return {
        "id": result.id,
        "lance_db_id": getattr(result, "lance_db_id", None),
        "account": account_value,
        "username": getattr(result, "username", account_value),
        "display_name": getattr(result, "display_name", result.profile_name),
        "profile_name": result.profile_name,
        "platform": platform_normalized,
        "platform_id": getattr(result, "platform_id", None),
        "followers": result.followers,
        "followers_formatted": _format_number(result.followers),
        "avg_engagement": _format_engagement(result.avg_engagement),
        "avg_engagement_raw": result.avg_engagement,
        "business_category_name": result.business_category_name,
        "business_address": result.business_address,
        "biography": result.biography,
        "profile_image_link": profile_image,
        "profile_image_url": profile_image,
        "profile_url": profile_url,
        "business_email": getattr(result, "business_email", ""),
        "email_address": getattr(result, "email_address", ""),
        "posts": _parse_posts(getattr(result, "posts_raw", "")),
        "is_personal_creator": getattr(result, "is_personal_creator", True),
        "individual_vs_org_score": getattr(result, "individual_vs_org_score", 0),
        "generational_appeal_score": getattr(result, "generational_appeal_score", 0),
        "professionalization_score": getattr(result, "professionalization_score", 0),
        "relationship_status_score": getattr(result, "relationship_status_score", 0),
        "bm25_fts_score": getattr(result, "bm25_fts_score", None),
        "cos_sim_profile": getattr(result, "cos_sim_profile", None),
        "cos_sim_posts": getattr(result, "cos_sim_posts", None),
        "combined_score": getattr(result, "combined_score", 0.0),
        "keyword_similarity": getattr(result, "keyword_similarity", None),
        "profile_similarity": getattr(result, "profile_similarity", None),
        "content_similarity": getattr(result, "content_similarity", None),
        "vector_similarity_score": getattr(result, "vector_similarity_score", None),
        "profile_fts_source": getattr(result, "profile_fts_source", None),
        "posts_fts_source": getattr(result, "posts_fts_source", None),
        "score_mode": getattr(result, "score_mode", "hybrid"),
        "similarity_explanation": getattr(result, "similarity_explanation", ""),
        "fit_score": getattr(result, "fit_score", None),
        "fit_rationale": getattr(result, "fit_rationale", None),
        "fit_error": getattr(result, "fit_error", None),
        "fit_prompt": getattr(result, "fit_prompt", None),
        "fit_raw_response": getattr(result, "fit_raw_response", None),
    }


@router.get("/username/{username}", response_model=UsernameSearchResponse)
async def get_creator_by_username(username: str, search_engine=Depends(get_search_engine)):
    sanitized = username.strip()
    if not sanitized:
        raise HTTPException(status_code=400, detail="Username is required")

    try:
        result = search_engine.get_creator_by_username(sanitized)
        if not result:
            raise HTTPException(status_code=404, detail=f"Creator '@{sanitized}' not found")
        return {"success": True, "result": result_to_dict(result)}
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Username lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="Username lookup failed") from exc


@router.post("/", response_model=SearchResponse)
async def search_creators(request: SearchRequest, search_engine=Depends(get_search_engine)):
    logger.info(
        "Search request | method=%s limit=%s query=%s",
        request.method,
        request.limit,
        request.query,
    )

    try:
        results = search_engine.search_creators_for_campaign(
            query=request.query,
            method=request.method,
            limit=request.limit,
            min_followers=request.min_followers,
            max_followers=request.max_followers,
            min_engagement=request.min_engagement,
            max_engagement=request.max_engagement,
            location=request.location,
            category=request.category,
            is_verified=request.is_verified,
            is_business_account=request.is_business_account,
            lexical_scope=request.lexical_scope,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Search failed") from exc

    payload = [result_to_dict(result) for result in results]
    return SearchResponse(
        success=True,
        results=payload,
        count=len(payload),
        query=request.query,
        method=request.method,
    )


@router.post("/similar", response_model=SearchResponse)
async def similar_creators(request: SimilarSearchRequest, search_engine=Depends(get_search_engine)):
    logger.info("Similar search | account=%s limit=%s", request.account, request.limit)

    try:
        results = search_engine.find_similar_creators(
            reference_account=request.account,
            limit=request.limit,
            min_followers=request.min_followers,
            max_followers=request.max_followers,
            min_engagement=request.min_engagement,
            max_engagement=request.max_engagement,
            location=request.location,
            category=request.category,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Similar search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Similar search failed") from exc

    payload = [result_to_dict(result) for result in results]
    return SearchResponse(
        success=True,
        results=payload,
        count=len(payload),
        query=request.account,
        method="similar",
    )


@router.post("/category", response_model=SearchResponse)
async def category_search(request: CategorySearchRequest, search_engine=Depends(get_search_engine)):
    logger.info(
        "Category search | category=%s location=%s limit=%s",
        request.category,
        request.location,
        request.limit,
    )

    try:
        results = search_engine.search_by_category(
            category=request.category,
            location=request.location,
            limit=request.limit,
            min_followers=request.min_followers,
            max_followers=request.max_followers,
            min_engagement=request.min_engagement,
            max_engagement=request.max_engagement,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Category search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Category search failed") from exc

    payload = [result_to_dict(result) for result in results]
    return SearchResponse(
        success=True,
        results=payload,
        count=len(payload),
        query=request.category,
        method="category",
    )


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_profiles(request: EvaluationRequest, search_engine=Depends(get_search_engine)):
    logger.info(
        "Evaluate profiles | brightdata=%s llm=%s count=%s",
        request.run_brightdata,
        request.run_llm,
        len(request.profiles),
    )

    try:
        results, debug = search_engine.evaluate_profiles(
            request.profiles,
            business_fit_query=request.business_fit_query,
            run_brightdata=request.run_brightdata,
            run_llm=request.run_llm,
            max_profiles=request.max_profiles,
            max_posts=request.max_posts,
            model=request.model,
            verbosity=request.verbosity,
            concurrency=request.concurrency,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Profile evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Profile evaluation failed") from exc

    payload = [result_to_dict(result) for result in results]
    return EvaluationResponse(
        success=True,
        results=payload,
        brightdata_results=debug.get("brightdata_results", []),
        profile_fit=debug.get("profile_fit", []),
        count=len(payload),
    )


@router.post("/evaluate/stream")
async def evaluate_profiles_stream(request: EvaluationRequest, search_engine=Depends(get_search_engine)):
    logger.info(
        "Evaluate profiles (stream) | brightdata=%s llm=%s count=%s",
        request.run_brightdata,
        request.run_llm,
        len(request.profiles),
    )

    def sse_event(stage: str, data: Dict[str, Any]) -> str:
        return f"data: {json.dumps({'stage': stage, 'data': data})}\n\n"

    def event_stream():
        queue: "SimpleQueue[Optional[Dict[str, Any]]]" = SimpleQueue()

        def progress(stage: str, data: Dict[str, Any]) -> None:
            queue.put({"stage": stage, "data": data})

        def worker() -> None:
            try:
                results, debug = search_engine.evaluate_profiles(
                    request.profiles,
                    business_fit_query=request.business_fit_query,
                    run_brightdata=request.run_brightdata,
                    run_llm=request.run_llm,
                    max_profiles=request.max_profiles,
                    max_posts=request.max_posts,
                    model=request.model,
                    verbosity=request.verbosity,
                    concurrency=request.concurrency,
                    progress_cb=progress,
                )
                payload = [result_to_dict(result) for result in results]
                queue.put(
                    {
                        "stage": "completed",
                        "data": {
                            "results": payload,
                            "brightdata_results": debug.get("brightdata_results", []),
                            "profile_fit": debug.get("profile_fit", []),
                            "count": len(payload),
                        },
                    }
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Streaming evaluation failed: %s", exc)
                queue.put({"stage": "error", "data": {"message": str(exc)}})
            finally:
                queue.put(None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = queue.get()
            if item is None:
                break
            yield sse_event(item["stage"], item["data"])

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
