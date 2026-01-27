"""Application settings and configuration."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    
    # Database
    database_url: str = "sqlite:///./data/aiasylum.db"
    
    # Security
    api_secret_key: str = "change-me-in-production"
    jwt_secret_key: str = "change-me-in-production"
    api_key_hmac_secret: str = "change-me-in-production"
    api_keys: str = ""  # Comma-separated
    
    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Request Limits
    max_request_size_mb: int = 10
    
    # Concurrency Control
    max_concurrent_workers: int = 5  # Maximum number of test runs that can execute simultaneously
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Deep Analysis
    enable_prompt_differential_analysis: bool = False
    enable_activation_patching: bool = False
    enable_cot_detection: bool = False
    
    # RAG
    enable_rag: bool = False
    ollama_embedding_model: str = "nomic-embed-text"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    @property
    def api_keys_list(self) -> List[str]:
        """Parse API keys into a list."""
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"
    
    @property
    def config_dir(self) -> Path:
        """Get config directory."""
        return self.project_root / "config"


# Global settings instance
settings = Settings()
