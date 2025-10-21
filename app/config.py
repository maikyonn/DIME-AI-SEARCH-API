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
    
    # Database settings
    DB_PATH: Optional[str] = None
    TEXT_DB_PATH: Optional[str] = None
    TABLE_NAME: str = "influencer_facets"
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    
    # Image refresh settings
    BRIGHTDATA_API_TOKEN: Optional[str] = None
    BRIGHTDATA_API_KEY: Optional[str] = None
    BRIGHTDATA_DATASET_ID: Optional[str] = None
    BRIGHTDATA_BASE_URL: Optional[str] = "https://api.brightdata.com/datasets/v3"
    BRIGHTDATA_POLL_INTERVAL: int = 30

    # OpenAI / LLM settings
    OPENAI_API_KEY: Optional[str] = None
    
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

def _resolve_default_db_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    primary_path = os.path.join(
        project_root,
        "DIME-AI-DB",
        "data",
        "lancedb",
    )
    legacy_path = os.path.join(project_root, "DIME-AI-DB", "influencers_vectordb")
    return primary_path if os.path.exists(primary_path) else legacy_path


def _resolve_default_text_db_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    primary_path = os.path.join(
        project_root,
        "DIME-AI-DB",
        "data",
        "lancedb",
    )
    legacy_path = os.path.join(project_root, "DIME-AI-DB", "influencers_lancedb")
    return primary_path if os.path.exists(primary_path) else legacy_path


# Set default DB path if not provided
if not settings.DB_PATH:
    settings.DB_PATH = _resolve_default_db_path()

if not settings.TEXT_DB_PATH:
    settings.TEXT_DB_PATH = _resolve_default_text_db_path()
