"""
FastAPI wrapper for the LanceDB facet search engine built on influencer_facets.
Provides higher-level orchestration for dense, lexical, and hybrid retrieval.
"""
import os
import sys
import json
from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from dataclasses import dataclass

# Add the DIME-AI-DB src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
dime_db_root = os.path.join(project_root, "DIME-AI-DB")
dime_db_src = os.path.join(dime_db_root, "src")

for path in (dime_db_root, dime_db_src):
    if path not in sys.path and os.path.isdir(path):
        sys.path.insert(0, path)

# Import from the new vector search engine
from .vector_search import VectorSearchEngine, SearchWeights, SearchParams
from app.core.post_filter import BrightDataClient, ProfileFitAssessor, ProfileFitResult
from app.config import settings

@dataclass
class SearchResult:
    """A single search result from the influencer database"""
    id: int
    account: str
    profile_name: str
    followers: int
    avg_engagement: float
    business_category_name: str
    business_address: str
    biography: str
    profile_image_link: str = ""
    profile_url: Optional[str] = None
    is_personal_creator: bool = True
    is_verified: Optional[bool] = None
    posts_raw: Optional[str] = None
    lance_db_id: Optional[str] = None
    platform: Optional[str] = None
    platform_id: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    profile_image_url: Optional[str] = None
    # Original database LLM score columns
    individual_vs_org_score: int = 0
    generational_appeal_score: int = 0
    professionalization_score: int = 0
    relationship_status_score: int = 0
    # Search score components
    bm25_fts_score: Optional[float] = None
    cos_sim_profile: Optional[float] = None
    cos_sim_posts: Optional[float] = None
    combined_score: float = 0.0
    # Vector similarity scores (direct vector comparison)
    keyword_similarity: Optional[float] = None
    profile_similarity: Optional[float] = None
    content_similarity: Optional[float] = None
    vector_similarity_score: Optional[float] = None
    similarity_explanation: str = ""
    score_mode: str = "hybrid"
    profile_fts_source: Optional[str] = None
    posts_fts_source: Optional[str] = None
    fit_score: Optional[int] = None
    fit_rationale: Optional[str] = None
    fit_error: Optional[str] = None
    fit_prompt: Optional[str] = None
    fit_raw_response: Optional[str] = None


class FastAPISearchEngine:
    """FastAPI wrapper for the VectorSearchEngine"""
    
    def __init__(self, db_path: str):
        self.engine = VectorSearchEngine(
            db_path=db_path,
            table_name=settings.TABLE_NAME or "influencer_facets",
            model_name=settings.EMBED_MODEL,
        )
        
        # Content categories for campaign matching
        self.content_categories = {
            'lifestyle': ['lifestyle', 'daily life', 'life', 'routine', 'vlog', 'personal', 'day in my life', 'grwm'],
            'fashion': ['fashion', 'style', 'outfit', 'ootd', 'clothing', 'trendy', 'streetwear', 'aesthetic'],
            'beauty': ['beauty', 'makeup', 'skincare', 'cosmetics', 'glam', 'tutorial', 'review', 'routine'],
            'tech': ['tech', 'technology', 'gadget', 'app', 'phone', 'gaming', 'review', 'unboxing'],
            'fitness': ['fitness', 'workout', 'gym', 'health', 'wellness', 'yoga', 'training', 'sport'],
            'travel': ['travel', 'trip', 'vacation', 'explore', 'adventure', 'destination', 'wanderlust'],
            'food': ['food', 'cooking', 'recipe', 'restaurant', 'foodie', 'chef', 'cuisine', 'dining'],
            'entertainment': ['music', 'dance', 'comedy', 'entertainment', 'performance', 'artist', 'creative']
        }
    
    def _convert_to_search_result(self, row) -> SearchResult:
        """Convert pandas row to SearchResult dataclass"""
        # Helper function to safely convert values, handling NaN
        def safe_int(value, default=0):
            if value is None or (isinstance(value, float) and str(value).lower() == 'nan'):
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
                
        def safe_float(value, default=0.0):
            if value is None or (isinstance(value, float) and str(value).lower() == 'nan'):
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        def safe_optional_float(value):
            if value is None or (isinstance(value, float) and str(value).lower() == 'nan'):
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        def safe_str(value):
            if value is None:
                return None
            text_value = str(value)
            return text_value if text_value.lower() != 'nan' else None

        lance_identifier = safe_str(row.get('lance_db_id'))
        platform_value = safe_str(row.get('platform'))

        def safe_bool(value):
            if isinstance(value, bool):
                return value
            if value is None:
                return None
            text_value = str(value).strip().lower()
            if text_value in {'true', '1', 'yes', 'y'}:
                return True
            if text_value in {'false', '0', 'no', 'n'}:
                return False
            return None

        lance_identifier = safe_str(row.get('lance_db_id'))
        platform_value = safe_str(row.get('platform'))
        account_value = safe_str(row.get('account') or row.get('username') or row.get('display_name')) or ""
        profile_name_value = safe_str(row.get('profile_name') or row.get('display_name') or row.get('username') or account_value) or ""
        avg_engagement_value = safe_float(row.get('avg_engagement', row.get('engagement_rate', 0.0)), 0.0)
        business_category = safe_str(row.get('business_category_name') or row.get('occupation') or '') or ''
        business_address = safe_str(row.get('business_address') or row.get('location') or '') or ''
        posts_raw_value = safe_str(row.get('posts') or row.get('posts_raw'))

        return SearchResult(
            id=safe_int(row.get('id', lance_identifier or 0)),
            account=account_value,
            profile_name=profile_name_value,
            followers=safe_int(row.get('followers', 0)),
            avg_engagement=avg_engagement_value,
            business_category_name=business_category,
            business_address=business_address,
            biography=str(row.get('biography', '') or row.get('profile_text') or ''),
            profile_image_link=str(row.get('profile_image_link') or row.get('profile_image_url') or ''),
            profile_url=safe_str(row.get('profile_url') or row.get('url')),
            is_personal_creator=bool(safe_int(row.get('individual_vs_org_score', 5)) < 5),
            is_verified=safe_bool(row.get('is_verified')),
            posts_raw=posts_raw_value,
            lance_db_id=lance_identifier,
            platform=platform_value.lower() if isinstance(platform_value, str) else platform_value,
            platform_id=safe_str(row.get('platform_id')),
            username=safe_str(row.get('username') or row.get('account')),
            display_name=safe_str(row.get('display_name') or row.get('profile_name') or row.get('full_name')),
            profile_image_url=safe_str(row.get('profile_image_link') or row.get('profile_image_url')),
            # Original database LLM score columns (keep as integers)
            individual_vs_org_score=safe_int(row.get('individual_vs_org_score', 0)),
            generational_appeal_score=safe_int(row.get('generational_appeal_score', 0)),
            professionalization_score=safe_int(row.get('professionalization_score', 0)),
            relationship_status_score=safe_int(row.get('relationship_status_score', 0)),
            # Search score components
            bm25_fts_score=safe_optional_float(row.get('bm25_fts_score')),
            cos_sim_profile=safe_optional_float(row.get('cos_sim_profile')),
            cos_sim_posts=safe_optional_float(row.get('cos_sim_posts')),
            combined_score=safe_float(row.get('combined_score', row.get('vector_similarity_score', 0.0))),
            # Vector similarity scores (direct vector comparison)
            keyword_similarity=safe_optional_float(row.get('keyword_similarity')),
            profile_similarity=safe_optional_float(row.get('profile_similarity')),
            content_similarity=safe_optional_float(row.get('content_similarity')),
            vector_similarity_score=safe_optional_float(row.get('vector_similarity_score')),
            similarity_explanation=str(row.get('similarity_explanation', '')),
            score_mode=(safe_str(row.get('score_mode')) or 'hybrid'),
            profile_fts_source=safe_str(row.get('profile_fts_source')),
            posts_fts_source=safe_str(row.get('posts_fts_source')),
            fit_score=None,
            fit_rationale=None,
            fit_error=None,
            fit_prompt=None,
            fit_raw_response=None
        )

    def _coerce_search_result(self, payload: Union[SearchResult, Dict[str, Any]]) -> SearchResult:
        """Accept either API payloads or in-process SearchResult instances."""
        if isinstance(payload, SearchResult):
            return payload
        if isinstance(payload, dict):
            return self._convert_to_search_result(payload)
        raise TypeError(f"Unsupported profile payload type: {type(payload)!r}")

    def _prepare_results(
        self,
        profiles: List[Union[SearchResult, Dict[str, Any]]],
        max_profiles: Optional[int] = None,
    ) -> List[SearchResult]:
        if not profiles:
            return []

        all_results = [self._coerce_search_result(payload) for payload in profiles]
        if max_profiles is None:
            limit_count = len(all_results)
        else:
            limit_count = max(1, min(int(max_profiles), len(all_results)))

        return all_results[:limit_count]
        
    def search_creators_for_campaign(
        self,
        *,
        query: str,
        method: str = "hybrid",
        limit: int = 20,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None,
        min_engagement: Optional[float] = None,
        max_engagement: Optional[float] = None,
        location: Optional[str] = None,
        category: Optional[str] = None,
        is_verified: Optional[bool] = None,
        is_business_account: Optional[bool] = None,
        lexical_scope: str = "bio",
    ) -> List[SearchResult]:
        """Run a single-pass search with predictable behaviour."""

        method_lower = (method or "").strip().lower()

        query_text = (query or "").strip()
        if not query_text:
            return []

        filters: Dict[str, Any] = {}

        follower_lower = int(min_followers) if min_followers is not None else None
        follower_upper = int(max_followers) if max_followers is not None else None
        if follower_lower is not None or follower_upper is not None:
            filters["followers"] = (
                follower_lower if follower_lower is not None else 0,
                follower_upper,
            )

        eng_lower = float(min_engagement) if min_engagement is not None else None
        eng_upper = float(max_engagement) if max_engagement is not None else None
        if eng_lower is not None or eng_upper is not None:
            filters["engagement_rate"] = (
                eng_lower if eng_lower is not None else 0.0,
                eng_upper,
            )

        if is_verified is not None:
            filters["is_verified"] = is_verified

        if is_business_account is not None:
            filters["is_business_account"] = is_business_account

        if location:
            filters["location"] = location.strip()

        if category:
            filters["business_category_name"] = category.strip()

        params = SearchParams(
            query=query_text,
            method=method_lower,
            limit=max(1, limit),
            filters=filters or None,
            lexical_include_posts=(method_lower == "lexical" and lexical_scope == "bio_posts"),
        )

        results_df = self.engine.search(params=params)

        search_results: List[SearchResult] = []
        for _, row in results_df.iterrows():
            search_results.append(self._convert_to_search_result(row))

        for item in search_results:
            item.score_mode = method_lower or "hybrid"
            if method_lower == "lexical":
                item.cos_sim_profile = None
                item.cos_sim_posts = None
                item.vector_similarity_score = None
                item.keyword_similarity = None
                item.profile_similarity = None
                item.content_similarity = None
            elif method_lower == "semantic":
                item.bm25_fts_score = None
                item.profile_fts_source = None
                item.posts_fts_source = None

        return search_results

    def evaluate_profiles(
        self,
        profiles: List[Union[SearchResult, Dict[str, Any]]],
        *,
        business_fit_query: Optional[str] = None,
        run_brightdata: bool = False,
        run_llm: bool = False,
        max_profiles: Optional[int] = None,
        max_posts: int = 6,
        model: str = "gpt-5-mini",
        verbosity: str = "medium",
        concurrency: int = 64,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Run optional BrightData refresh and/or LLM scoring on a result set."""

        search_results = self._prepare_results(profiles, max_profiles)

        if not search_results:
            return [], {"brightdata_results": [], "profile_fit": []}

        debug: Dict[str, Any] = {
            "brightdata_results": [],
            "profile_fit": [],
        }

        if progress_cb:
            progress_cb(
                "evaluation_started",
                {"count": len(search_results), "run_brightdata": run_brightdata, "run_llm": run_llm},
            )

        if run_llm:
            if not business_fit_query:
                raise ValueError("business_fit_query is required when run_llm is True")

            search_results, fit_debug = self.run_profile_fit_stage(
                search_results,
                business_fit_query=business_fit_query,
                max_profiles=len(search_results),
                concurrency=concurrency,
                max_posts=max_posts,
                model=model,
                verbosity=verbosity,
                use_brightdata=run_brightdata,
                progress_cb=progress_cb,
            )
            debug.update(fit_debug)
            return search_results, debug

        if run_brightdata:
            search_results, brightdata_debug = self.run_brightdata_stage(
                search_results,
                max_profiles=len(search_results),
                progress_cb=progress_cb,
            )
            debug.update(brightdata_debug)

        return search_results, debug

    def run_brightdata_stage(
        self,
        profiles: List[Union[SearchResult, Dict[str, Any]]],
        *,
        max_profiles: Optional[int] = None,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        search_results = self._prepare_results(profiles, max_profiles)
        debug: Dict[str, Any] = {"brightdata_results": []}

        if not search_results:
            return search_results, debug

        if progress_cb:
            progress_cb("brightdata_started", {"count": len(search_results)})
        try:
            debug["brightdata_results"] = self._refresh_profiles_with_brightdata(search_results)
        except Exception as exc:  # pylint: disable=broad-except
            debug["brightdata_error"] = str(exc)
            if progress_cb:
                progress_cb("brightdata_completed", {"count": 0, "error": str(exc)})
        else:
            if progress_cb:
                progress_cb(
                    "brightdata_completed",
                    {"count": len(debug["brightdata_results"])},
                )
        return search_results, debug

    def run_profile_fit_stage(
        self,
        profiles: List[Union[SearchResult, Dict[str, Any]]],
        *,
        business_fit_query: str,
        max_profiles: Optional[int] = None,
        concurrency: int = 64,
        max_posts: int = 6,
        model: str = "gpt-5-mini",
        verbosity: str = "medium",
        use_brightdata: bool = False,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        if not business_fit_query:
            raise ValueError("business_fit_query must be provided for profile fit stage")

        search_results = self._prepare_results(profiles, max_profiles)
        if not search_results:
            return search_results, {"brightdata_results": [], "profile_fit": []}

        fitted_results, debug = self._apply_profile_fit(
            search_results,
            business_fit_query=business_fit_query,
            limit=len(search_results),
            concurrency=concurrency,
            max_posts=max_posts,
            model=model,
            verbosity=verbosity,
            use_brightdata=use_brightdata,
            progress_cb=progress_cb,
        )
        return fitted_results, debug

    def _apply_profile_fit(
        self,
        results: List[SearchResult],
        *,
        business_fit_query: str,
        limit: int,
        concurrency: int,
        max_posts: int,
        model: str,
        verbosity: str,
        use_brightdata: bool,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Run stage-two LLM scoring and re-rank results."""
        if not results:
            return results, {"brightdata_results": [], "profile_fit": []}

        limit = max(1, min(limit, len(results))) if limit else len(results)
        subset = results[:limit]
        remainder = results[limit:] if limit < len(results) else []

        brightdata_records: List[Dict[str, Any]] = []
        if use_brightdata:
            if progress_cb:
                progress_cb("brightdata_started", {"count": len(subset)})
            try:
                brightdata_records = self._refresh_profiles_with_brightdata(subset)
            except Exception as exc:  # pylint: disable=broad-except
                print(f'[WARN] BrightData refresh failed: {exc}')
                if progress_cb:
                    progress_cb(
                        "brightdata_completed",
                        {"count": 0, "error": str(exc)},
                    )
            else:
                if progress_cb:
                    progress_cb(
                        "brightdata_completed",
                        {"count": len(brightdata_records)},
                    )

        documents = []
        for item in subset:
            documents.append(
                {
                    "account": item.account,
                    "profile_url": item.profile_url or (f"https://instagram.com/{item.account}" if item.account else None),
                    "followers": item.followers,
                    "biography": item.biography,
                    "profile_name": item.profile_name,
                    "business_category_name": item.business_category_name,
                    "category_name": item.business_category_name,
                    "is_verified": getattr(item, 'is_verified', None),
                    "posts": item.posts_raw,
                }
            )

        assessor = ProfileFitAssessor(
            business_query=business_fit_query,
            model=model,
            verbosity=verbosity,
            max_posts=max_posts,
            concurrency=concurrency,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        fit_results = assessor.score_profiles(documents)
        total = len(fit_results)
        fit_map: Dict[str, ProfileFitResult] = {}
        profile_fit_debug: List[Dict[str, Any]] = []
        for idx, fit in enumerate(fit_results, start=1):
            key = (fit.account or '').lower() if fit.account else None
            if key:
                fit_map[key] = fit
            elif fit.profile_url:
                fit_map[fit.profile_url.lower()] = fit

            profile_fit_debug.append(
                {
                    "account": fit.account,
                    "profile_url": fit.profile_url,
                    "followers": fit.followers,
                    "score": fit.score,
                    "rationale": fit.rationale,
                    "error": fit.error,
                    "prompt": fit.prompt,
                    "raw_response": fit.raw_response,
                }
            )
            if progress_cb:
                progress_cb("fit_progress", {"completed": idx, "total": total, "account": fit.account})

        for item in subset:
            fit: Optional[ProfileFitResult] = None
            account_key = item.account.lower() if item.account else None
            if account_key and account_key in fit_map:
                fit = fit_map[account_key]
            elif item.profile_url and item.profile_url.lower() in fit_map:
                fit = fit_map[item.profile_url.lower()]

            if fit:
                item.fit_score = fit.score
                item.fit_rationale = fit.rationale
                item.fit_error = fit.error
                item.fit_prompt = fit.prompt
                item.fit_raw_response = fit.raw_response
            else:
                item.fit_score = None
                item.fit_rationale = None
                item.fit_error = None
                item.fit_prompt = None
                item.fit_raw_response = None

        if progress_cb:
            progress_cb("fit_completed", {"total": total})

        scored_subset = sorted(
            subset,
            key=lambda r: ((r.fit_score or 0), r.combined_score),
            reverse=True,
        )

        debug_payload = {
            "brightdata_results": brightdata_records,
            "profile_fit": profile_fit_debug,
        }

        return scored_subset + remainder, debug_payload

    def _refresh_profiles_with_brightdata(self, profiles: List[SearchResult]) -> List[Dict[str, Any]]:
        """Fetch latest profile data from BrightData and update in-place."""
        urls: List[str] = []
        for item in profiles:
            url = item.profile_url or (f"https://instagram.com/{item.account}" if item.account else None)
            if url:
                urls.append(url)

        if not urls:
            return []

        client = BrightDataClient()
        dataframe = client.fetch_profiles(urls)
        records = json.loads(dataframe.to_json(orient="records")) if not dataframe.empty else []
        profile_map = BrightDataClient.dataframe_to_profile_map(dataframe)

        for item in profiles:
            candidates: List[str] = []
            if item.profile_url:
                candidates.append(item.profile_url.lower())
            if item.account:
                candidates.append(f"https://instagram.com/{item.account}".lower())
                candidates.append(item.account.lower())

            match = None
            for key in candidates:
                if key in profile_map:
                    match = profile_map[key]
                    break

            if not match:
                continue

            biography = match.get('biography') or match.get('bio')
            if biography:
                item.biography = str(biography)

            followers = match.get('followers') or match.get('followers_count')
            if followers is not None:
                try:
                    item.followers = int(followers)
                except (TypeError, ValueError):
                    pass

            posts_value = match.get('posts') or match.get('posts_json')
            if posts_value:
                item.posts_raw = posts_value

        return records

    def run_profile_fit_preview(
        self,
        *,
        business_fit_query: str,
        account: Optional[str] = None,
        profile_url: Optional[str] = None,
        max_posts: int = 6,
        model: str = "gpt-5-mini",
        verbosity: str = "medium",
        use_brightdata: bool = False,
        concurrency: int = 2,
    ) -> ProfileFitResult:
        """Score a single profile against a business brief."""
        if not business_fit_query:
            raise ValueError("business_fit_query is required")

        profile: Optional[SearchResult] = None
        if account:
            profile = self.get_creator_by_username(account)

        if profile is None and profile_url:
            profile = self._get_profile_by_url(profile_url)

        if profile is None:
            raise ValueError("Profile not found for profile fit preview")

        profiles = [profile]
        if use_brightdata:
            try:
                self._refresh_profiles_with_brightdata(profiles)
            except Exception as exc:  # pylint: disable=broad-except
                print(f'[WARN] BrightData refresh failed: {exc}')

        assessor = ProfileFitAssessor(
            business_query=business_fit_query,
            model=model,
            verbosity=verbosity,
            max_posts=max_posts,
            concurrency=max(1, concurrency),
            openai_api_key=settings.OPENAI_API_KEY,
        )

        documents = [
            {
                "account": profile.account,
                "profile_url": profile.profile_url or (f"https://instagram.com/{profile.account}" if profile.account else None),
                "followers": profile.followers,
                "biography": profile.biography,
                "profile_name": profile.profile_name,
                "business_category_name": profile.business_category_name,
                "category_name": profile.business_category_name,
                "is_verified": profile.is_verified,
                "posts": profile.posts_raw,
            }
        ]

        fit_result = assessor.score_profiles(documents)[0]
        profile.fit_score = fit_result.score
        profile.fit_rationale = fit_result.rationale
        profile.fit_error = fit_result.error
        return fit_result

    def _get_profile_by_url(self, profile_url: str) -> Optional[SearchResult]:
        """Fetch a single creator profile by profile_url."""
        if not profile_url:
            return None

        normalized = profile_url.strip().replace("'", "''")
        if not normalized:
            return None

        profile_row = self.engine.get_profile_by_url(normalized)
        if profile_row is None:
            return None

        row = profile_row.copy()
        row['account'] = row.get('username') or row.get('account') or ''
        row['profile_name'] = row.get('display_name') or row.get('profile_name') or row.get('username') or ''
        row.setdefault('bm25_fts_score', row.get('keyword_score'))
        row.setdefault('cos_sim_profile', row.get('profile_score'))
        row.setdefault('cos_sim_posts', row.get('content_score'))
        row.setdefault('profile_fts_source', row.get('profile_text'))
        row.setdefault('posts_fts_source', row.get('posts_text'))
        row.setdefault('combined_score', row.get('cos_sim_profile'))
        row.setdefault('vector_similarity_score', row.get('combined_score'))
        return self._convert_to_search_result(row)

    def get_creator_by_username(self, username: str) -> Optional[SearchResult]:
        """Fetch a single creator profile by username."""
        if not username:
            return None

        normalized = username.strip().lstrip('@')
        if not normalized:
            return None

        profile_row = self.engine.get_profile_by_username(normalized)
        if profile_row is None or getattr(profile_row, 'empty', False):
            return None

        row = profile_row.copy()
        row['account'] = row.get('username') or normalized
        row['profile_name'] = row.get('display_name') or row.get('username') or normalized
        row.setdefault('bm25_fts_score', row.get('keyword_score'))
        row.setdefault('cos_sim_profile', row.get('profile_score'))
        row.setdefault('cos_sim_posts', row.get('content_score'))
        row.setdefault('profile_fts_source', row.get('profile_text'))
        row.setdefault('posts_fts_source', row.get('posts_text'))
        row.setdefault('combined_score', row.get('cos_sim_profile'))
        row.setdefault('vector_similarity_score', row.get('combined_score'))
        return self._convert_to_search_result(row)
    
    def match_creators_to_business(
        self,
        business_description: str,
        method: str = "hybrid",
        limit: int = 20,
        min_followers: Optional[int] = 1000,
        max_followers: Optional[int] = 10000000,
        min_engagement: float = 0.0,
        location: Optional[str] = None,
        target_category: Optional[str] = None,
    ) -> List[SearchResult]:
        """Match creators to a business brief using the simplified search pipeline."""

        search_query = self._business_to_creator_query(business_description, target_category)

        return self.search_creators_for_campaign(
            query=search_query,
            method=method,
            limit=limit,
            min_followers=min_followers,
            max_followers=max_followers,
            min_engagement=min_engagement,
            location=location,
            category=target_category,
        )
    
    def _business_to_creator_query(self, business_description: str, target_category: Optional[str] = None) -> str:
        """Convert business description to creator search query"""
        query_parts = [business_description]
        
        # Add category-specific terms
        if target_category and target_category in self.content_categories:
            category_terms = self.content_categories[target_category][:3]
            query_parts.extend(category_terms)
        
        # Add influencer/creator context
        query_parts.append("content creator influencer")
        
        return " ".join(query_parts)
    
    def find_similar_creators(
        self,
        reference_account: str,
        limit: int = 10,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None,
        min_engagement: Optional[float] = None,
        max_engagement: Optional[float] = None,
        location: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        """Find creators similar to a reference account using vector similarity."""

        filters: Dict[str, Any] = {}

        follower_lower = int(min_followers) if min_followers is not None else None
        follower_upper = int(max_followers) if max_followers is not None else None
        if follower_lower is not None or follower_upper is not None:
            filters["followers"] = (
                follower_lower if follower_lower is not None else 0,
                follower_upper,
            )

        eng_lower = float(min_engagement) if min_engagement is not None else None
        eng_upper = float(max_engagement) if max_engagement is not None else None
        if eng_lower is not None or eng_upper is not None:
            filters["engagement_rate"] = (
                eng_lower if eng_lower is not None else 0.0,
                eng_upper,
            )

        if location:
            filters["location"] = location.strip()

        if category:
            filters["business_category_name"] = category.strip()

        results_df = self.engine.search_similar_by_vectors(
            account_name=reference_account,
            limit=limit,
            weights=SearchWeights(keyword=0.2, profile=0.5, content=0.3),
            filters=filters or None,
        )

        search_results: List[SearchResult] = []
        for _, row in results_df.iterrows():
            search_results.append(self._convert_to_search_result(row))

        return search_results
    
    def search_by_category(
        self,
        category: str,
        location: Optional[str] = None,
        limit: int = 15,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None,
        min_engagement: Optional[float] = None,
        max_engagement: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search creators by category with sensible defaults."""

        query_parts = [category]
        if category in self.content_categories:
            query_parts.extend(self.content_categories[category][:3])
        if location:
            query_parts.append(location)

        query = " ".join(query_parts)

        return self.search_creators_for_campaign(
            query=query,
            method="hybrid",
            limit=limit,
            min_followers=min_followers,
            max_followers=max_followers,
            min_engagement=min_engagement,
            max_engagement=max_engagement,
            location=location,
            category=category,
        )
