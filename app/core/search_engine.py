"""
FastAPI wrapper for the vector search engine.
This module provides a clean interface to the new three-vector search functionality.
"""
import os
import sys
import pandas as pd
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Add the DIME-AI-DB src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
dime_db_src = os.path.join(project_root, "DIME-AI-DB", "src")
sys.path.insert(0, dime_db_src)

# Import from the new vector search engine
from search.vector_search import VectorSearchEngine, SearchWeights
from app.core.post_filter import BrightDataClient, ProfileFitAssessor, ProfileFitResult

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
    # Original database LLM score columns
    individual_vs_org_score: int = 0
    generational_appeal_score: int = 0
    professionalization_score: int = 0
    relationship_status_score: int = 0
    # Vector search similarity scores (text-based search)
    keyword_score: float = 0.0
    profile_score: float = 0.0
    content_score: float = 0.0
    combined_score: float = 0.0
    # Vector similarity scores (direct vector comparison)
    keyword_similarity: float = 0.0
    profile_similarity: float = 0.0
    content_similarity: float = 0.0
    vector_similarity_score: float = 0.0
    similarity_explanation: str = ""
    fit_score: Optional[int] = None
    fit_rationale: Optional[str] = None
    fit_error: Optional[str] = None


class FastAPISearchEngine:
    """FastAPI wrapper for the VectorSearchEngine"""
    
    def __init__(self, db_path: str):
        self.engine = VectorSearchEngine(
            db_path=db_path,
            table_name="influencer_profiles"
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
        
        def safe_str(value):
            if value is None:
                return None
            text_value = str(value)
            return text_value if text_value.lower() != 'nan' else None

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

        return SearchResult(
            id=safe_int(row.get('id', row.get('lance_db_id', 0))),
            account=str(row.get('account', '')),
            profile_name=str(row.get('profile_name', '')),
            followers=safe_int(row.get('followers', 0)),
            avg_engagement=safe_float(row.get('avg_engagement', 0.0)),
            business_category_name=str(row.get('business_category_name', '')),
            business_address=str(row.get('business_address', '')),
            biography=str(row.get('biography', '')),
            profile_image_link=str(row.get('profile_image_link', '')),
            profile_url=safe_str(row.get('profile_url') or row.get('url')),
            is_personal_creator=bool(safe_int(row.get('individual_vs_org_score', 5)) < 5),
            is_verified=safe_bool(row.get('is_verified')),
            posts_raw=safe_str(row.get('posts')),
            # Original database LLM score columns (keep as integers)
            individual_vs_org_score=safe_int(row.get('individual_vs_org_score', 0)),
            generational_appeal_score=safe_int(row.get('generational_appeal_score', 0)),
            professionalization_score=safe_int(row.get('professionalization_score', 0)),
            relationship_status_score=safe_int(row.get('relationship_status_score', 0)),
            # Vector search similarity scores (text-based search)
            keyword_score=safe_float(row.get('keyword_score', 0.0)),
            profile_score=safe_float(row.get('profile_score', 0.0)),
            content_score=safe_float(row.get('content_score', 0.0)),
            combined_score=safe_float(row.get('combined_score', 0.0)),
            # Vector similarity scores (direct vector comparison)
            keyword_similarity=safe_float(row.get('keyword_similarity', 0.0)),
            profile_similarity=safe_float(row.get('profile_similarity', 0.0)),
            content_similarity=safe_float(row.get('content_similarity', 0.0)),
            vector_similarity_score=safe_float(row.get('vector_similarity_score', 0.0)),
            similarity_explanation=str(row.get('similarity_explanation', '')),
            fit_score=None,
            fit_rationale=None,
            fit_error=None
        )
        
    def search_creators_for_campaign(
        self,
        query: str,
        method: str = "hybrid",
        limit: int = 20,
        min_followers: int = 0,
        max_followers: Optional[int] = None,
        min_engagement: float = 0.0,
        max_engagement: Optional[float] = None,
        location_filter: Optional[str] = None,
        target_category: Optional[str] = None,
        relevance_keywords: Optional[List[str]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        similarity_threshold: Optional[float] = None,
        return_vectors: bool = False,
        business_fit_query: Optional[str] = None,
        post_filter_limit: Optional[int] = None,
        post_filter_concurrency: int = 8,
        post_filter_max_posts: int = 6,
        post_filter_model: str = "gpt-5-mini",
        post_filter_verbosity: str = "medium",
        post_filter_use_brightdata: bool = False,
        # Account Type Filters
        is_verified: Optional[bool] = None,
        is_business_account: Optional[bool] = None,
        # LLM Score Filters
        min_individual_vs_org_score: Optional[int] = None,
        max_individual_vs_org_score: Optional[int] = None,
        min_generational_appeal_score: Optional[int] = None,
        max_generational_appeal_score: Optional[int] = None,
        min_professionalization_score: Optional[int] = None,
        max_professionalization_score: Optional[int] = None,
        min_relationship_status_score: Optional[int] = None,
        max_relationship_status_score: Optional[int] = None,
        # Content Stats Filters
        min_posts_count: Optional[int] = None,
        max_posts_count: Optional[int] = None,
        min_following: Optional[int] = None,
        max_following: Optional[int] = None
    ) -> List[SearchResult]:
        """Search creators for a campaign using vector search"""
        
        # Build enhanced query with keywords
        enhanced_query = query
        if relevance_keywords:
            enhanced_query += " " + " ".join(relevance_keywords)
        if target_category and target_category in self.content_categories:
            category_keywords = " ".join(self.content_categories[target_category][:5])
            enhanced_query += " " + category_keywords
        
        # Convert custom weights to SearchWeights
        weights = None
        if custom_weights:
            weights = SearchWeights(
                keyword=custom_weights.get('keyword', 0.33),
                profile=custom_weights.get('profile', 0.33),
                content=custom_weights.get('content', 0.34)
            )
        elif method == "keyword":
            weights = SearchWeights(keyword=0.7, profile=0.2, content=0.1)
        elif method == "profile":
            weights = SearchWeights(keyword=0.3, profile=0.6, content=0.1)
        elif method == "content":
            weights = SearchWeights(keyword=0.2, profile=0.3, content=0.5)
        
        # Build filters with enhanced validation
        filters = {}
        
        # Handle follower filters with flexible ranges
        if max_followers is not None:
            filters['followers'] = (int(min_followers), int(max_followers))
        elif min_followers > 0:
            filters['followers'] = (int(min_followers), 100000000)  # Large upper bound
        
        # Handle engagement filters with flexible ranges
        if min_engagement is not None or max_engagement is not None:
            min_eng = float(min_engagement) if min_engagement is not None else 0.0
            max_eng = float(max_engagement) if max_engagement is not None else 1.0
            if min_eng > 0 or max_eng < 1.0:
                filters['engagement_rate'] = (min_eng, max_eng)
        
        # Handle account type filters
        if is_verified is not None:
            filters['is_verified'] = is_verified
        if is_business_account is not None:
            filters['is_business_account'] = is_business_account
        
        # Handle content stats filters
        if min_posts_count is not None or max_posts_count is not None:
            min_posts = int(min_posts_count) if min_posts_count is not None else 0
            max_posts = int(max_posts_count) if max_posts_count is not None else 100000
            filters['posts_count'] = (min_posts, max_posts)
        
        if min_following is not None or max_following is not None:
            min_fol = int(min_following) if min_following is not None else 0
            max_fol = int(max_following) if max_following is not None else 100000000
            filters['following'] = (min_fol, max_fol)
        
        # Handle LLM score filters
        if min_individual_vs_org_score is not None and max_individual_vs_org_score is not None:
            filters['individual_vs_org_score'] = (int(min_individual_vs_org_score), int(max_individual_vs_org_score))
        elif min_individual_vs_org_score is not None:
            filters['individual_vs_org_score'] = (int(min_individual_vs_org_score), 10)
        elif max_individual_vs_org_score is not None:
            filters['individual_vs_org_score'] = (0, int(max_individual_vs_org_score))
            
        if min_generational_appeal_score is not None and max_generational_appeal_score is not None:
            filters['generational_appeal_score'] = (int(min_generational_appeal_score), int(max_generational_appeal_score))
        elif min_generational_appeal_score is not None:
            filters['generational_appeal_score'] = (int(min_generational_appeal_score), 10)
        elif max_generational_appeal_score is not None:
            filters['generational_appeal_score'] = (0, int(max_generational_appeal_score))
            
        if min_professionalization_score is not None and max_professionalization_score is not None:
            filters['professionalization_score'] = (int(min_professionalization_score), int(max_professionalization_score))
        elif min_professionalization_score is not None:
            filters['professionalization_score'] = (int(min_professionalization_score), 10)
        elif max_professionalization_score is not None:
            filters['professionalization_score'] = (0, int(max_professionalization_score))
            
        if min_relationship_status_score is not None and max_relationship_status_score is not None:
            filters['relationship_status_score'] = (int(min_relationship_status_score), int(max_relationship_status_score))
        elif min_relationship_status_score is not None:
            filters['relationship_status_score'] = (int(min_relationship_status_score), 10)
        elif max_relationship_status_score is not None:
            filters['relationship_status_score'] = (0, int(max_relationship_status_score))
        
        # Perform search
        results_df = self.engine.search(
            query=enhanced_query,
            limit=limit,
            weights=weights,
            filters=filters if filters else None
        )
        
        # Convert to SearchResult objects
        search_results = []
        for _, row in results_df.iterrows():
            search_results.append(self._convert_to_search_result(row))

        if business_fit_query:
            fit_limit = post_filter_limit or len(search_results)
            try:
                search_results = self._apply_profile_fit(
                    search_results,
                    business_fit_query=business_fit_query,
                    limit=fit_limit,
                    concurrency=post_filter_concurrency,
                    max_posts=post_filter_max_posts,
                    model=post_filter_model,
                    verbosity=post_filter_verbosity,
                    use_brightdata=post_filter_use_brightdata,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f'[WARN] Post-filter stage failed: {exc}')

        return search_results

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
    ) -> List[SearchResult]:
        """Run stage-two LLM scoring and re-rank results."""
        if not results:
            return results

        limit = max(1, min(limit, len(results))) if limit else len(results)
        subset = results[:limit]
        remainder = results[limit:] if limit < len(results) else []

        if use_brightdata:
            try:
                self._refresh_profiles_with_brightdata(subset)
            except Exception as exc:  # pylint: disable=broad-except
                print(f'[WARN] BrightData refresh failed: {exc}')

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
        )

        fit_results = assessor.score_profiles(documents)
        fit_map: Dict[str, ProfileFitResult] = {}
        for fit in fit_results:
            key = (fit.account or '').lower() if fit.account else None
            if key:
                fit_map[key] = fit
            elif fit.profile_url:
                fit_map[fit.profile_url.lower()] = fit

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
            else:
                item.fit_score = None
                item.fit_rationale = None
                item.fit_error = None

        scored_subset = sorted(
            subset,
            key=lambda r: ((r.fit_score or 0), r.combined_score),
            reverse=True,
        )

        return scored_subset + remainder

    def _refresh_profiles_with_brightdata(self, profiles: List[SearchResult]) -> None:
        """Fetch latest profile data from BrightData and update in-place."""
        urls: List[str] = []
        for item in profiles:
            url = item.profile_url or (f"https://instagram.com/{item.account}" if item.account else None)
            if url:
                urls.append(url)

        if not urls:
            return

        client = BrightDataClient()
        dataframe = client.fetch_profiles(urls)
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

        self.engine.connect()
        table = getattr(self.engine, 'table', None)
        if table is None:
            return None

        queries = [
            f"profile_url = '{normalized}'",
            f"LOWER(profile_url) = '{normalized.lower()}'",
        ]

        for query in queries:
            try:
                results = table.search().where(query).to_pandas()
            except Exception:
                continue

            if not results.empty:
                row = results.iloc[0]
                return self._convert_to_search_result(row)

        return None

    def get_creator_by_username(self, username: str) -> Optional[SearchResult]:
        """Fetch a single creator profile by username."""
        if not username:
            return None

        normalized = username.strip().lstrip('@')
        if not normalized:
            return None

        # Escape single quotes to avoid query syntax issues
        sanitized = normalized.replace("'", "''")

        # Ensure the underlying table connection is available
        self.engine.connect()
        table = getattr(self.engine, 'table', None)
        if table is None:
            return None

        # Try exact match first, then case-insensitive, then partial match
        queries = [
            f"account = '{sanitized}'",
            f"LOWER(account) = '{sanitized.lower()}'",
            f"LOWER(account) LIKE '%{sanitized.lower()}%'"
        ]

        for query in queries:
            try:
                results = table.search().where(query).to_pandas()
            except Exception:
                continue

            if not results.empty:
                # Prefer the exact match order by placing the most relevant row first
                row = results.iloc[0]
                return self._convert_to_search_result(row)

        return None
    
    def match_creators_to_business(
        self,
        business_description: str,
        method: str = "hybrid",
        limit: int = 20,
        custom_weights: Optional[Dict[str, float]] = None,
        min_followers: int = 1000,
        max_followers: int = 10000000,
        min_engagement: float = 0.0,
        location_filter: Optional[str] = None,
        target_category: Optional[str] = None
    ) -> List[SearchResult]:
        """Match creators to business description using vector search"""
        
        # Transform business description into creator search query
        search_query = self._business_to_creator_query(business_description, target_category)
        
        return self.search_creators_for_campaign(
            query=search_query,
            method=method,
            limit=limit,
            min_followers=min_followers,
            max_followers=max_followers,
            min_engagement=min_engagement,
            location_filter=location_filter,
            target_category=target_category,
            custom_weights=custom_weights
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
        min_followers: int = 0,
        max_followers: Optional[int] = None,
        min_engagement: Optional[float] = None,
        max_engagement: Optional[float] = None,
        location_filter: Optional[str] = None,
        target_category: Optional[str] = None,
        similarity_threshold: float = 0.1,
        use_vector_similarity: bool = True,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """Find similar creators to a reference account using advanced vector similarity"""
        
        # Build filters with enhanced validation
        filters = {}
        
        # Handle follower filters
        if max_followers is not None:
            filters['followers'] = (int(min_followers), int(max_followers))
        elif min_followers > 0:
            filters['followers'] = (int(min_followers), 100000000)
        
        # Handle engagement filters
        if min_engagement is not None or max_engagement is not None:
            min_eng = float(min_engagement) if min_engagement is not None else 0.0
            max_eng = float(max_engagement) if max_engagement is not None else 1.0
            if min_eng > 0 or max_eng < 1.0:
                filters['engagement_rate'] = (min_eng, max_eng)
        
        # Convert custom weights if provided
        weights = SearchWeights(keyword=0.4, profile=0.4, content=0.2)
        if custom_weights:
            weights = SearchWeights(
                keyword=custom_weights.get('keyword', 0.4),
                profile=custom_weights.get('profile', 0.4),
                content=custom_weights.get('content', 0.2)
            )
        
        # Use the new vector-based similarity search by default
        if use_vector_similarity:
            results_df = self.engine.search_similar_by_vectors(
                account_name=reference_account,
                limit=limit,
                weights=weights,
                similarity_threshold=similarity_threshold,
                include_similarity_scores=True,
                filters=filters if filters else None
            )
        else:
            # Fallback to legacy text-based search
            results_df = self.engine.search_similar_profiles(
                account_name=reference_account,
                limit=limit,
                weights=weights
            )
            
            # Apply filters manually for legacy method if needed
            if filters and not results_df.empty:
                if 'followers' in filters:
                    min_fol, max_fol = filters['followers']
                    results_df = results_df[(results_df['followers'] >= min_fol) & (results_df['followers'] <= max_fol)]
                if 'engagement_rate' in filters:
                    min_eng, max_eng = filters['engagement_rate']
                    results_df = results_df[(results_df['avg_engagement'] >= min_eng) & (results_df['avg_engagement'] <= max_eng)]
        
        # Convert to SearchResult objects
        search_results = []
        for _, row in results_df.iterrows():
            search_results.append(self._convert_to_search_result(row))
        
        return search_results
    
    def search_by_category(
        self,
        category: str,
        location: Optional[str] = None,
        limit: int = 15,
        min_followers: int = 0,
        max_followers: Optional[int] = None,
        min_engagement: Optional[float] = None,
        max_engagement: Optional[float] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search creators by category using vector search"""
        
        # Build search query from category
        query = category
        if category in self.content_categories:
            category_keywords = " ".join(self.content_categories[category][:5])
            query = f"{category} {category_keywords}"
        
        if location:
            query += f" {location}"
        
        # Build filters with enhanced validation
        filters = {}
        
        # Handle follower filters
        if max_followers is not None:
            filters['followers'] = (int(min_followers), int(max_followers))
        elif min_followers > 0:
            filters['followers'] = (int(min_followers), 100000000)
        
        # Handle engagement filters
        if min_engagement is not None or max_engagement is not None:
            min_eng = float(min_engagement) if min_engagement is not None else 0.0
            max_eng = float(max_engagement) if max_engagement is not None else 1.0
            if min_eng > 0 or max_eng < 1.0:
                filters['engagement_rate'] = (min_eng, max_eng)
        
        # Convert custom weights or use category-optimized defaults
        weights = SearchWeights(keyword=0.4, profile=0.5, content=0.1)
        if custom_weights:
            weights = SearchWeights(
                keyword=custom_weights.get('keyword', 0.4),
                profile=custom_weights.get('profile', 0.5),
                content=custom_weights.get('content', 0.1)
            )
        
        # Perform search with profile focus for category matching
        results_df = self.engine.search(
            query=query,
            limit=limit,
            weights=weights,
            filters=filters if filters else None
        )
        
        # Convert to SearchResult objects
        search_results = []
        for _, row in results_df.iterrows():
            search_results.append(self._convert_to_search_result(row))
        
        return search_results
