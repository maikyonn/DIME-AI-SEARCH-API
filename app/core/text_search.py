from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

import lancedb

from app.config import settings
from app.core.search_engine import FastAPISearchEngine, SearchResult


class TextSearchEngine:
    """Perform plaintext biography searches across the influencer facets dataset."""

    def __init__(
        self,
        *,
        table_path: str,
        table_name: str = "influencer_facets",
        vector_engine: Optional[FastAPISearchEngine] = None,
    ) -> None:
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Text LanceDB table not found: {table_path}")

        self.table_path = table_path
        self.db = lancedb.connect(table_path)
        self.table = self.db.open_table(table_name)
        self.vector_engine = vector_engine or FastAPISearchEngine(settings.DB_PATH)

    def search_biography(
        self,
        *,
        query: str,
        limit: int = 50,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None,
        min_engagement: Optional[float] = None,
    ) -> List[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        predicates: List[str] = ["content_type = 'profile'"]
        if min_followers is not None:
            predicates.append(f"followers >= {int(min_followers)}")
        if max_followers is not None:
            predicates.append(f"followers <= {int(max_followers)}")
        if min_engagement is not None:
            try:
                eng_threshold = float(min_engagement) / 100.0
                predicates.append(f"engagement_rate >= {eng_threshold}")
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

        search = self.table.search(query)
        if predicates:
            search = search.where(" AND ".join(predicates))

        try:
            df = search.limit(max(1, limit)).to_pandas()
        except Exception:
            df = pd.DataFrame()

        if df.empty and self.vector_engine is not None:
            vector_engine = getattr(self.vector_engine, "engine", None)
            if vector_engine is not None:
                vector_engine._ensure_profiles_loaded()  # type: ignore[attr-defined]
                profiles_df = getattr(vector_engine, "_profiles_df", None)
                if profiles_df is not None and not profiles_df.empty:
                    candidates = profiles_df.copy()
                    mask = candidates['biography'].astype(str).str.contains(query, case=False, na=False)
                    if min_followers is not None:
                        mask &= candidates['followers'].fillna(0) >= int(min_followers)
                    if max_followers is not None:
                        mask &= candidates['followers'].fillna(0) <= int(max_followers)
                    if min_engagement is not None:
                        try:
                            eng_threshold = float(min_engagement) / 100.0
                            mask &= candidates['engagement_rate'].fillna(0.0) >= eng_threshold
                        except (TypeError, ValueError):  # pragma: no cover - defensive
                            pass
                    df = candidates[mask].head(max(1, limit)).copy()

        if df.empty:
            return []

        results: List[SearchResult] = []
        for _, row in df.iterrows():
            results.append(self.vector_engine._convert_to_search_result(row))  # type: ignore[attr-defined]

        results.sort(
            key=lambda r: (
                (r.fit_score if r.fit_score is not None else -1),
                r.combined_score if r.combined_score is not None else 0.0,
            ),
            reverse=True,
        )
        return results


__all__ = ["TextSearchEngine"]
