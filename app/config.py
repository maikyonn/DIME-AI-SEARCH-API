import os
from typing import Optional, List, Union
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    APP_NAME: str = "GenZ Creator Search API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    PORT: int = 7001
    
    # Database settings
    DB_PATH: Optional[str] = None
    TEXT_DB_PATH: Optional[str] = None
    TABLE_NAME: str = "influencer_facets"

    # API settings
    API_V1_PREFIX: str = "/search"
    VIEWER_PORT: int = 7002
    VIEWER_ROOT_PATH: str = "/db-viewer"

    # Image refresh settings
    BRIGHTDATA_MAX_URLS: int = 50
    BRIGHTDATA_SERVICE_URL: str = "http://localhost:7100/brightdata/images"
    BRIGHTDATA_JOB_TIMEOUT: int = 600
    BRIGHTDATA_JOB_POLL_INTERVAL: int = 5

    # OpenAI / LLM settings
    OPENAI_API_KEY: Optional[str] = None

    # DeepInfra embeddings
    DEEPINFRA_API_KEY: Optional[str] = None
    DEEPINFRA_ENDPOINT: str = "https://api.deepinfra.com/v1/openai"
    EMBED_MODEL: str = "google/embeddinggemma-300m"

    # CORS settings
    ALLOWED_ORIGINS: Union[str, List[str]] = ["*"]
    
    @field_validator('ALLOWED_ORIGINS', mode='before')
    @classmethod
    def parse_origins(cls, v):
        """Parse ALLOWED_ORIGINS from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def _candidate_db_roots() -> List[str]:
    """Return possible DIME-AI-DB locations (env override, sibling repo, nested copy)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(current_dir))
    workspace_root = os.path.dirname(repo_root)

    env_root = os.getenv("DIME_AI_DB_ROOT") or os.getenv("DIME_DB_ROOT")
    candidates: List[str] = []
    if env_root:
        candidates.append(env_root)
    candidates.extend(
        [
            os.path.join(repo_root, "DIME-AI-DB"),
            os.path.join(workspace_root, "DIME-AI-DB"),
        ]
    )

    # Deduplicate while preserving order
    seen = set()
    unique_candidates: List[str] = []
    for path in candidates:
        norm = os.path.abspath(path)
        if norm not in seen:
            seen.add(norm)
            unique_candidates.append(norm)
    return unique_candidates or [repo_root]


def _resolve_default_db_path() -> str:
    """Prefer the new LanceDB directory, fall back to legacy vectordb."""
    for root in _candidate_db_roots():
        candidate = os.path.join(root, "data", "lancedb")
        if os.path.exists(candidate):
            return candidate

    for root in _candidate_db_roots():
        legacy = os.path.join(root, "influencers_vectordb")
        if os.path.exists(legacy):
            return legacy

    # Last resort: point to the expected modern layout even if missing
    first_root = _candidate_db_roots()[0]
    return os.path.join(first_root, "data", "lancedb")


def _resolve_default_text_db_path() -> str:
    """Prefer the consolidated LanceDB directory, fall back to historical text DB."""
    for root in _candidate_db_roots():
        candidate = os.path.join(root, "data", "lancedb")
        if os.path.exists(candidate):
            return candidate

    for root in _candidate_db_roots():
        legacy = os.path.join(root, "influencers_lancedb")
        if os.path.exists(legacy):
            return legacy

    first_root = _candidate_db_roots()[0]
    return os.path.join(first_root, "data", "lancedb")


# Set default DB path if not provided
if not settings.DB_PATH:
    settings.DB_PATH = _resolve_default_db_path()

if not settings.TEXT_DB_PATH:
    settings.TEXT_DB_PATH = _resolve_default_text_db_path()

if not settings.EMBED_MODEL:
    settings.EMBED_MODEL = "google/embeddinggemma-300m"
