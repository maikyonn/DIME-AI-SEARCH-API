from __future__ import annotations

import os
from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import lancedb
import numpy as np
import pandas as pd
from openai import OpenAI

from app.config import settings

LOGGER = logging.getLogger(__name__)

DEFAULT_DEEPINFRA_ENDPOINT = "https://api.deepinfra.com/v1/openai"


def _format_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "NULL"
    text = str(value).replace("'", "''")
    return f"'{text}'"


@dataclass
class SearchWeights:
    """Weighting scheme for search aggregation."""

    keyword: float = 0.20  # lexical/FTS contribution
    profile: float = 0.40  # profile facet embedding
    content: float = 0.40  # posts facet embedding

    def normalised(self) -> "SearchWeights":
        total = self.keyword + self.profile + self.content
        if total <= 0:
            return SearchWeights(keyword=0.0, profile=0.5, content=0.5)
        return SearchWeights(
            keyword=self.keyword / total,
            profile=self.profile / total,
            content=self.content / total,
        )


@dataclass
class SearchParams:
    """Parameters for running a search against LanceDB."""

    query: str
    method: str = "hybrid"
    limit: int = 20
    weights: Optional[SearchWeights] = None
    filters: Optional[Dict[str, Any]] = None
    lexical_include_posts: bool = False


class VectorSearchEngine:
    """Wrapper around the LanceDB facet table for dense + lexical search."""

    PROFILE_FACET = "profile"
    POSTS_FACET = "posts"

    def __init__(
        self,
        *,
        db_path: str,
        table_name: str = "influencer_facets",
        model_name: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.table_name = table_name
        self.model_name = model_name or settings.EMBED_MODEL or os.environ.get("EMBED_MODEL", "google/embeddinggemma-300m")

        self.db: Optional[lancedb.db.DBConnection] = None
        self.table: Optional[lancedb.table.Table] = None
        self.embedder: Optional["DeepInfraQueryEmbedder"] = None

        self._profiles_df: Optional[pd.DataFrame] = None
        self._profile_columns: Tuple[str, ...] = tuple()

        self.connect()
        self._ensure_profiles_loaded()
        self._init_embedder()

    def _init_embedder(self) -> None:
        api_key = settings.DEEPINFRA_API_KEY or os.environ.get("DEEPINFRA_API_KEY")
        endpoint = settings.DEEPINFRA_ENDPOINT or os.environ.get("DEEPINFRA_ENDPOINT", DEFAULT_DEEPINFRA_ENDPOINT)

        if not api_key:
            LOGGER.warning(
                "DeepInfra API key not configured; semantic and hybrid searches will be unavailable."
            )
            self.embedder = None
            return

        try:
            self.embedder = DeepInfraQueryEmbedder(
                model_name=self.model_name,
                api_key=api_key,
                endpoint=endpoint or DEFAULT_DEEPINFRA_ENDPOINT,
            )
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.warning("Failed to initialise DeepInfra embedder: %s", exc)
            self.embedder = None

    # ---------------------------------------------------------------------
    # Connection + caching helpers
    # ---------------------------------------------------------------------
    def connect(self) -> None:
        if self.db is not None and self.table is not None:
            return

        if not os.path.isdir(self.db_path):
            raise FileNotFoundError(f"LanceDB path not found: {self.db_path}")

        self.db = lancedb.connect(self.db_path)
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception as exc:  # pragma: no cover - Lance errors bubble
            raise RuntimeError(
                f"Failed to open LanceDB table '{self.table_name}'"
            ) from exc

    def refresh(self) -> None:
        self._profiles_df = None
        self._profile_columns = tuple()
        self._ensure_profiles_loaded()

    def profile_count(self) -> int:
        self._ensure_profiles_loaded()
        if self._profiles_df is None:
            return 0
        return len(self._profiles_df)

    def _ensure_profiles_loaded(self) -> None:
        if self._profiles_df is not None:
            return

        assert self.table is not None
        dataframe = self.table.to_pandas()
        if dataframe.empty:
            self._profiles_df = pd.DataFrame()
            self._profile_columns = tuple()
            return

        profiles = dataframe[dataframe["content_type"] == self.PROFILE_FACET].copy()
        # Drop heavy columns not needed for metadata caching
        profiles = profiles.drop(
            columns=["sparse_indices", "sparse_values"], errors="ignore"
        )
        # Ensure consistent index for lookups
        if "lance_db_id" in profiles.columns:
            profiles = profiles.set_index("lance_db_id", drop=False)

        self._profiles_df = profiles
        self._profile_columns = tuple(c for c in profiles.columns if c not in {"content_type"})

    # ---------------------------------------------------------------------
    # Public search APIs
    # ---------------------------------------------------------------------
    def search(
        self,
        params: Optional[SearchParams] = None,
        *,
        query: Optional[str] = None,
        limit: int = 20,
        weights: Optional[SearchWeights] = None,
        filters: Optional[Dict[str, Any]] = None,
        method: str = "hybrid",
    ) -> pd.DataFrame:
        if params is None:
            params = SearchParams(
                query=query or "",
                method=method,
                limit=limit,
                weights=weights,
                filters=filters,
            )
        else:
            if query is not None:
                raise ValueError("Provide either SearchParams or keyword arguments, not both")

        method_lower = (params.method or "hybrid").strip().lower()
        resolved_limit = max(1, params.limit or 20)
        include_semantic = method_lower in {"semantic", "hybrid"}
        include_lexical = method_lower in {"lexical", "hybrid"}

        # Resolve default weights per mode
        if method_lower == "lexical":
            resolved_weights = SearchWeights(keyword=1.0, profile=0.0, content=0.0)
        elif method_lower == "semantic":
            resolved_weights = params.weights or SearchWeights(keyword=0.0, profile=0.6, content=0.4)
        else:
            resolved_weights = params.weights or SearchWeights(keyword=0.35, profile=0.4, content=0.25)

        vector = None
        query_text = (params.query or "").strip()
        if include_semantic and query_text:
            if self.embedder is None:
                raise ValueError(
                    "Semantic search requires DeepInfra embeddings. "
                    "Set DEEPINFRA_API_KEY (and optionally DEEPINFRA_ENDPOINT) to enable semantic or hybrid modes."
                )
            vector = self._encode_query(query_text)

        return self._run_search(
            vector=vector,
            query_text=query_text,
            limit=resolved_limit,
            weights=resolved_weights,
            filters=params.filters,
            include_semantic=include_semantic,
            include_lexical=include_lexical,
            lexical_include_posts=params.lexical_include_posts,
        )

    def search_with_vector(
        self,
        *,
        vector: np.ndarray,
        limit: int = 20,
        weights: Optional[SearchWeights] = None,
        filters: Optional[Dict[str, Any]] = None,
        method: str = "semantic",
    ) -> pd.DataFrame:
        method_lower = (method or "semantic").strip().lower()
        include_semantic = method_lower in {"semantic", "hybrid"}
        include_lexical = method_lower in {"lexical", "hybrid"}

        if method_lower == "lexical":
            resolved_weights = SearchWeights(keyword=1.0, profile=0.0, content=0.0)
        elif method_lower == "semantic":
            resolved_weights = weights or SearchWeights(keyword=0.0, profile=0.6, content=0.4)
        else:
            resolved_weights = weights or SearchWeights(keyword=0.35, profile=0.4, content=0.25)

        return self._run_search(
            vector=vector if include_semantic else None,
            query_text=None,
            limit=max(1, limit),
            weights=resolved_weights,
            filters=filters,
            include_semantic=include_semantic,
            include_lexical=include_lexical,
            lexical_include_posts=False,
        )

    def search_similar_by_vectors(
        self,
        *,
        account_name: str,
        limit: int = 20,
        weights: Optional[SearchWeights] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if not account_name:
            return pd.DataFrame()

        self._ensure_profiles_loaded()
        lance_id = self._find_lance_id(account_name)
        if lance_id is None:
            return pd.DataFrame()

        profile_row = self._get_profile_row(lance_id)
        if profile_row is None or "embedding" not in profile_row:
            return pd.DataFrame()

        vector = np.asarray(profile_row["embedding"], dtype=np.float32)
        results = self.search_with_vector(
            vector=vector,
            limit=limit + 1,  # allow exclusion of anchor
            weights=weights,
            filters=filters,
            method="semantic",
        )

        if results.empty:
            return results

        filtered = results[results["lance_db_id"].str.lower() != lance_id.lower()]
        return filtered.head(limit).reset_index(drop=True)

    def get_profile_by_username(self, username: str) -> Optional[pd.Series]:
        if not username:
            return None
        self._ensure_profiles_loaded()
        lance_id = self._find_lance_id(username)
        if lance_id is None:
            return None
        return self._get_profile_row(lance_id)

    def get_profile_by_url(self, profile_url: str) -> Optional[pd.Series]:
        if not profile_url:
            return None
        self._ensure_profiles_loaded()
        df = self._profiles_df
        if df is None or df.empty or "profile_url" not in df.columns:
            return None

        normalized = profile_url.strip().lower()
        matches = df[df["profile_url"].astype(str).str.lower() == normalized]
        if matches.empty:
            return None
        return matches.iloc[0]

    # ------------------------------------------------------------------
    # Internal core
    # ------------------------------------------------------------------
    def _run_search(
        self,
        *,
        vector: Optional[np.ndarray],
        query_text: Optional[str],
        limit: int,
        weights: Optional[SearchWeights],
        filters: Optional[Dict[str, Any]],
        include_semantic: bool,
        include_lexical: bool,
        lexical_include_posts: bool,
    ) -> pd.DataFrame:
        if self.table is None:
            raise RuntimeError("Vector search table is not initialised")

        self._ensure_profiles_loaded()
        weights = (weights or SearchWeights()).normalised()
        gather_limit = max(1, limit)

        filter_expr = self._build_filter_expression(filters)

        profile_dense = pd.DataFrame()
        posts_dense = pd.DataFrame()
        if include_semantic and vector is not None:
            profile_dense = self._search_dense(vector, self.PROFILE_FACET, gather_limit, filter_expr)
            posts_dense = self._search_dense(vector, self.POSTS_FACET, gather_limit, filter_expr)

        lexical_df = pd.DataFrame()
        if include_lexical and query_text:
            lexical_df = self._search_lexical(query_text, gather_limit, filter_expr, include_posts=lexical_include_posts)

        entries: Dict[str, Dict[str, Any]] = {}

        self._accumulate_dense(entries, profile_dense, facet=self.PROFILE_FACET)
        self._accumulate_dense(entries, posts_dense, facet=self.POSTS_FACET)
        self._accumulate_lexical(entries, lexical_df)

        if not entries:
            return pd.DataFrame()

        lexical_only = include_lexical and not include_semantic

        records = self._finalise_entries(entries, weights, limit, lexical_only=lexical_only)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records)
        drop_columns = ["profile_vector_id", "posts_vector_id"]
        df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Lance query helpers
    # ------------------------------------------------------------------
    def _search_dense(
        self,
        vector: Optional[np.ndarray],
        content_type: str,
        limit: int,
        filter_expr: str,
    ) -> pd.DataFrame:
        if self.table is None or vector is None:
            return pd.DataFrame()

        search = (
            self.table.search(vector, vector_column_name="embedding")
            .metric("cosine")
        )
        condition = f"content_type = '{content_type}'"
        if filter_expr:
            condition = f"{condition} AND ({filter_expr})"
        search = search.where(condition)
        return search.limit(limit).to_pandas()

    def _search_lexical(
        self,
        query_text: str,
        limit: int,
        filter_expr: str,
        *,
        include_posts: bool = False,
    ) -> pd.DataFrame:
        if self.table is None:
            return pd.DataFrame()

        frames = []

        def run_query(content_type: str) -> pd.DataFrame:
            search = self.table.search(query_text)
            condition = f"content_type = '{content_type}'"
            if filter_expr:
                condition = f"{condition} AND ({filter_expr})"
            try:
                df = search.where(condition).limit(limit).to_pandas()
            except Exception:
                return pd.DataFrame()
            if not df.empty and "content_type" not in df.columns:
                df = df.assign(content_type=content_type)
            return df

        profile_df = run_query(self.PROFILE_FACET)
        if not profile_df.empty:
            frames.append(profile_df)

        if include_posts:
            posts_df = run_query(self.POSTS_FACET)
            if not posts_df.empty:
                frames.append(posts_df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _accumulate_dense(
        self,
        entries: Dict[str, Dict[str, Any]],
        df: pd.DataFrame,
        *,
        facet: str,
    ) -> None:
        if df.empty:
            return

        for _, row in df.iterrows():
            lance_id = str(row.get("lance_db_id") or "").strip()
            if not lance_id:
                continue

            entry = self._ensure_entry(entries, lance_id, row)
            if entry is None:
                continue

            distance = float(row.get("_distance", 1.0))
            similarity = max(0.0, 1.0 - distance)

            if facet == self.PROFILE_FACET and similarity > entry.get("profile_similarity", 0.0):
                entry["profile_similarity"] = similarity
                entry["profile_vector_id"] = row.get("vector_id")
                entry["profile_text"] = row.get("text")
            elif facet == self.POSTS_FACET and similarity > entry.get("posts_similarity", 0.0):
                entry["posts_similarity"] = similarity
                entry["posts_vector_id"] = row.get("vector_id")
                entry["posts_text"] = row.get("text")

    def _accumulate_lexical(
        self,
        entries: Dict[str, Dict[str, Any]],
        df: pd.DataFrame,
    ) -> None:
        if df.empty:
            return

        for _, row in df.iterrows():
            lance_id = str(row.get("lance_db_id") or "").strip()
            if not lance_id:
                continue

            entry = self._ensure_entry(entries, lance_id, row)
            if entry is None:
                continue

            score = float(row.get("_score", 0.0))
            entry["lexical_score_raw"] = max(entry.get("lexical_score_raw", 0.0), score)

            content_type = str(row.get("content_type") or "").lower()
            text_value = row.get("text")
            if content_type == self.PROFILE_FACET:
                if text_value:
                    entry["profile_text"] = text_value
            elif content_type == self.POSTS_FACET:
                if text_value:
                    entry["posts_text"] = text_value

    def _finalise_entries(
        self,
        entries: Dict[str, Dict[str, Any]],
        weights: SearchWeights,
        limit: int,
        *,
        lexical_only: bool = False,
    ) -> list[Dict[str, Any]]:
        if not entries:
            return []

        max_lexical = max((entry.get("lexical_score_raw", 0.0) for entry in entries.values()), default=0.0)
        records: list[Dict[str, Any]] = []

        for lance_id, entry in entries.items():
            profile_sim = float(entry.get("profile_similarity", 0.0))
            posts_sim = float(entry.get("posts_similarity", 0.0))
            lexical_raw = float(entry.get("lexical_score_raw", 0.0))
            lexical_norm = lexical_raw / max_lexical if max_lexical > 0 else 0.0

            if lexical_only and lexical_raw <= 0.0:
                continue

            combined = (
                weights.profile * profile_sim
                + weights.content * posts_sim
                + weights.keyword * lexical_norm
            )

            record = {
                **{k: entry.get(k) for k in self._profile_columns if k not in {"embedding", "vector_id", "text"}},
                "lance_db_id": lance_id,
                "account": entry.get("username") or lance_id,
                "profile_name": entry.get("display_name") or entry.get("username") or lance_id,
                "profile_similarity": profile_sim,
                "posts_similarity": posts_sim,
                "content_similarity": posts_sim,
                "bm25_fts_score": lexical_raw,
                "keyword_similarity": lexical_norm,
                "cos_sim_profile": profile_sim,
                "cos_sim_posts": posts_sim,
                "combined_score": combined,
                "vector_similarity_score": combined,
                "similarity_explanation": "",
                "posts": entry.get("posts_text"),
                "posts_raw": entry.get("posts_text"),
                "profile_fts_source": entry.get("profile_text"),
                "posts_fts_source": entry.get("posts_text"),
            }

            # Ensure followers and other numeric fields default to sensible values
            if "followers" not in record:
                record["followers"] = 0

            records.append(record)

        records.sort(key=lambda item: item.get("combined_score", 0.0), reverse=True)
        return records[: max(1, limit)]

    def _ensure_entry(
        self,
        entries: Dict[str, Dict[str, Any]],
        lance_id: str,
        seed_row: Optional[pd.Series],
    ) -> Optional[Dict[str, Any]]:
        existing = entries.get(lance_id)
        if existing is not None:
            return existing

        profile_row = self._get_profile_row(lance_id)
        source = profile_row if profile_row is not None else seed_row
        if source is None:
            return None

        if isinstance(source, pd.Series):
            data = source.to_dict()
        else:
            data = dict(source)

        entry: Dict[str, Any] = {}
        for column in self._profile_columns:
            if column in {"vector_id", "text"}:
                continue
            entry[column] = data.get(column)

        entry.setdefault("profile_text", data.get("text"))
        entry.setdefault("profile_vector_id", data.get("vector_id"))
        entry.setdefault("profile_similarity", 0.0)
        entry.setdefault("posts_similarity", 0.0)
        entry.setdefault("lexical_score_raw", 0.0)
        entries[lance_id] = entry
        return entry

    def _get_profile_row(self, lance_id: str) -> Optional[pd.Series]:
        if not lance_id or self._profiles_df is None:
            return None
        try:
            return self._profiles_df.loc[lance_id]
        except KeyError:
            return None

    def _build_filter_expression(self, filters: Optional[Dict[str, Any]]) -> str:
        if not filters:
            return ""

        valid_columns = set(self._profile_columns)
        clauses = []
        for key, constraint in filters.items():
            if key not in valid_columns:
                continue
            if isinstance(constraint, tuple) and len(constraint) == 2:
                lower, upper = constraint
                if lower is not None:
                    clauses.append(f"{key} >= {_format_literal(lower)}")
                if upper is not None:
                    clauses.append(f"{key} <= {_format_literal(upper)}")
            else:
                if constraint is None:
                    continue
                clauses.append(f"{key} = {_format_literal(constraint)}")

        return " AND ".join(clauses)

    def _find_lance_id(self, account_name: str) -> Optional[str]:
        lookup = account_name.strip().lstrip("@").lower()
        if not lookup or self._profiles_df is None:
            return None

        df = self._profiles_df
        for column in ("username", "display_name"):
            if column in df.columns:
                matches = df[df[column].astype(str).str.lower() == lookup]
                if not matches.empty:
                    return str(matches.iloc[0].get("lance_db_id"))
        return None

    def _encode_query(self, query: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("DeepInfra embedder not initialised")
        return self.embedder.embed(query)


class DeepInfraQueryEmbedder:
    """Lightweight client for DeepInfra's OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        endpoint: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError("DeepInfra API key is required for semantic search.")

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=(endpoint or DEFAULT_DEEPINFRA_ENDPOINT).rstrip("/"),
        )

    def embed(self, text: str) -> np.ndarray:
        payload = (text or "").strip()
        if not payload:
            raise ValueError("Cannot embed an empty query.")

        response = self.client.embeddings.create(
            model=self.model_name,
            input=[payload],
            encoding_format="float",
        )

        items = response.data or []
        if not items:
            raise RuntimeError("DeepInfra embedding response contained no vectors.")

        vector = np.asarray(items[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm


__all__ = ["VectorSearchEngine", "SearchWeights", "SearchParams"]
