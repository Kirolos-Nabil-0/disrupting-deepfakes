"""
Configuration management for the Deepfake Protection API.

This module handles all application settings using Pydantic Settings
with environment variable support.
"""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Settings
    app_name: str = Field(default="Deepfake Protection API", env="APP_NAME")
    version: str = Field(default="1.0.0", env="VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Security Settings
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS Settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="ALLOWED_ORIGINS"
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        env="ALLOWED_HOSTS" 
    )
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # Model Settings
    models_dir: str = Field(default="./models", env="MODELS_DIR")
    preload_models: bool = Field(default=True, env="PRELOAD_MODELS")
    cache_models: bool = Field(default=True, env="CACHE_MODELS")
    
    # GPU Settings
    device: str = Field(default="auto", env="DEVICE")  # auto, cpu, cuda, cuda:0, etc.
    max_gpu_memory_gb: Optional[float] = Field(default=None, env="MAX_GPU_MEMORY_GB")
    
    # Image Processing Settings
    max_image_size_mb: int = Field(default=10, env="MAX_IMAGE_SIZE_MB")
    max_image_dimension: int = Field(default=2048, env="MAX_IMAGE_DIMENSION")
    supported_formats: List[str] = Field(
        default=["jpeg", "jpg", "png", "webp"],
        env="SUPPORTED_FORMATS"
    )
    
    # Rate Limiting Settings
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Batch Processing Settings
    max_batch_size: int = Field(default=10, env="MAX_BATCH_SIZE")
    batch_timeout_seconds: int = Field(default=300, env="BATCH_TIMEOUT_SECONDS")
    
    # Redis Settings (for caching and task queue)
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Celery Settings
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Monitoring Settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Storage Settings
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    temp_dir: str = Field(default="./temp", env="TEMP_DIR")
    
    @validator("allowed_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("allowed_hosts", pre=True) 
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("supported_formats", pre=True)
    def parse_supported_formats(cls, v):
        """Parse supported formats from string or list."""
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(",")]
        return v
    
    @validator("device")
    def validate_device(cls, v):
        """Validate device setting."""
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @validator("models_dir", "upload_dir", "temp_dir")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins based on environment."""
        if self.debug:
            return self.allowed_origins + ["*"]
        return self.allowed_origins


class DatabaseSettings(BaseSettings):
    """Database settings (if needed for future features)."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    db_echo: bool = Field(default=False, env="DB_ECHO")
    db_pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    This function uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.
    """
    return Settings()


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """Get cached database settings."""
    return DatabaseSettings()


# Create environment template file
ENV_TEMPLATE = """
# API Settings
APP_NAME=Deepfake Protection API
VERSION=1.0.0
DEBUG=false

# Server Settings
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Security Settings
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
ALLOWED_HOSTS=localhost,127.0.0.1

# Logging Settings
LOG_LEVEL=INFO
LOG_FORMAT=json

# Model Settings
MODELS_DIR=./models
PRELOAD_MODELS=true
CACHE_MODELS=true

# GPU Settings
DEVICE=auto
MAX_GPU_MEMORY_GB=8

# Image Processing Settings
MAX_IMAGE_SIZE_MB=10
MAX_IMAGE_DIMENSION=2048
SUPPORTED_FORMATS=jpeg,jpg,png,webp

# Rate Limiting Settings
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10

# Batch Processing Settings
MAX_BATCH_SIZE=10
BATCH_TIMEOUT_SECONDS=300

# Redis Settings
REDIS_URL=redis://localhost:6379
REDIS_DB=0
REDIS_PASSWORD=

# Celery Settings
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Monitoring Settings
ENABLE_METRICS=true
METRICS_PORT=9090

# Storage Settings
UPLOAD_DIR=./uploads
TEMP_DIR=./temp
""".strip()


def create_env_file(path: str = ".env") -> None:
    """Create a .env file template if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(ENV_TEMPLATE)