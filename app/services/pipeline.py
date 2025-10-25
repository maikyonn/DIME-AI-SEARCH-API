"""Service that orchestrates the staged search → BrightData → LLM pipeline."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from app.core.search_engine import FastAPISearchEngine, SearchResult
from app.models.search import SearchPipelineRequest

ProgressCallback = Optional[Callable[[str, Dict[str, object]], None]]


class SearchPipelineService:
    """Run creator discovery plus optional enrichment in clearly defined stages."""

    def __init__(self, search_engine: FastAPISearchEngine) -> None:
        self._engine = search_engine

    def run_pipeline(
        self,
        request: SearchPipelineRequest,
        *,
        progress_cb: ProgressCallback = None,
    ) -> Tuple[List[SearchResult], Dict[str, object]]:
        """Execute the configured stages and emit progress events."""

        def emit(stage: str, data: Dict[str, object]) -> None:
            if progress_cb:
                progress_cb(stage, data)

        search_req = request.search
        emit(
            "search_started",
            {
                "query": search_req.query,
                "method": search_req.method,
                "limit": search_req.limit,
            },
        )

        search_results = self._engine.search_creators_for_campaign(
            query=search_req.query,
            method=search_req.method,
            limit=search_req.limit,
            min_followers=search_req.min_followers,
            max_followers=search_req.max_followers,
            min_engagement=search_req.min_engagement,
            max_engagement=search_req.max_engagement,
            location=search_req.location,
            category=search_req.category,
            is_verified=search_req.is_verified,
            is_business_account=search_req.is_business_account,
            lexical_scope=search_req.lexical_scope,
        )

        emit(
            "search_completed",
            {
                "count": len(search_results),
                "results": search_results,
            },
        )

        # Respect evaluation limits even if optional stages are disabled
        evaluation_inputs: List[SearchResult] = list(search_results)
        if request.max_profiles is not None and evaluation_inputs:
            limit = max(1, min(request.max_profiles, len(evaluation_inputs)))
            evaluation_inputs = evaluation_inputs[:limit]

        if not request.run_brightdata and not request.run_llm:
            emit(
                "evaluation_skipped",
                {"count": len(evaluation_inputs)},
            )
            return evaluation_inputs, {"brightdata_results": [], "profile_fit": []}

        results, debug = self._engine.evaluate_profiles(
            evaluation_inputs,
            business_fit_query=request.business_fit_query,
            run_brightdata=request.run_brightdata,
            run_llm=request.run_llm,
            max_profiles=len(evaluation_inputs),
            max_posts=request.max_posts,
            model=request.model,
            verbosity=request.verbosity,
            concurrency=request.concurrency,
            progress_cb=progress_cb,
        )

        return results, debug


__all__ = ["SearchPipelineService"]
