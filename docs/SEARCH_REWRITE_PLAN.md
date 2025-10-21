# Search Rewrite Plan

## Goals
- Keep the three user-facing modes (`lexical`, `semantic`, `hybrid`).
- Make each mode map to a single deterministic search path with minimal hidden toggles.
- Ensure lexical mode never runs vector lookups and only returns text-matching profiles.
- Ensure semantic mode relies on embeddings only (profile + content) with fixed weights.
- Make hybrid mode combine the lexical and semantic outputs with predictable default weights.
- Reduce implicit behaviours (oversampling, return_all_results, ad-hoc filters) to a clear, documented set.

## Target Behaviour
- **Lexical**: run LanceDB BM25 search on profile biographies; score = `_score`; results ordered strictly by that score.
- **Semantic**: run cosine similarity search on profile and post embeddings; results ordered by weighted similarity (`profile` 0.6, `content` 0.4 by default).
- **Hybrid**: execute both lexical and semantic; compute final score = `lexical_weight * lexical_norm + profile_weight * profile_sim + content_weight * content_sim` with defaults `(0.35, 0.4, 0.25)`.

## Filters
- Support a minimal, consistent filter set: followers, engagement, location, category, verification flags.
- Translate filters into LanceDB expressions in one place before any search mode runs.

## Weights
- Keep `SearchWeights` but limit valid combinations: lexical-only ignores profile/content weights, semantic-only ignores lexical weight, hybrid uses all three.

## API Surface
- Introduce a `SearchParams` dataclass consumed by `VectorSearchEngine.search`.
- `FastAPISearchEngine.search_creators_for_campaign` builds `SearchParams` from the request and no longer mutates the query or performs post-filter LLM scoring by default (still available via explicit flag).

## Testing
- Update existing vector search tests to cover each mode with the simplified pipeline.
- Add regression checks asserting lexical mode never emits results with zero `_score`.
