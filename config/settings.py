"""Application settings and configuration."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
    ollama_request_timeout: int = 1800  # Seconds (default 30 min); large models can be slow
    
    # Weight surgery. Both are gitignored; a bf16 3B copy is roughly 6 GB.
    weights_runs_dir: str = "runs/weights"
    weights_models_dir: str = "models"

    # Database
    database_url: str = "sqlite:///./data/aiasylum.db"
    
    # Security
    api_secret_key: str = "change-me-in-production"
    jwt_secret_key: str = "change-me-in-production"
    api_key_hmac_secret: str = "change-me-in-production"
    api_keys: str = ""  # Comma-separated
    # Fail-closed auth on every route (see api/security.py). Off by default so a
    # laptop dev loop is unchanged; the remote session script always sets it.
    require_auth: bool = False
    # Secure cookies need HTTPS. Over an SSH tunnel the browser talks plain
    # http://localhost, so this stays False there; set True behind TLS.
    session_cookie_secure: bool = False
    
    # CORS
    # Allow both localhost and 127.0.0.1 in dev (browsers treat them as different origins)
    cors_origins: str = (
        "http://localhost:3000,http://localhost:8000,"
        "http://127.0.0.1:3000,http://127.0.0.1:8000"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Request Limits
    max_request_size_mb: int = 10
    
    # Concurrency Control
    max_concurrent_workers: int = 5  # Maximum number of test runs that can execute simultaneously
    # Optional isolated Python for benchmarks on newer model architectures.
    benchmark_python: Optional[str] = Field(default=None, validation_alias="AIASYLUM_BENCHMARK_PYTHON")
    
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
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"
    
    @property
    def config_dir(self) -> Path:
        """Get config directory."""
        return self.project_root / "config"

    @property
    def weights_runs_root(self) -> Path:
        """Where derived directions and steering sweeps are written."""
        return self.project_root / self.weights_runs_dir

    @property
    def weights_models_root(self) -> Path:
        """The only directory weight surgery is allowed to write a model into.

        Every output path is built from this root plus a validated slug, so a
        request can never name a destination of its own.
        """
        return self.project_root / self.weights_models_dir


# Global settings instance
settings = Settings()
