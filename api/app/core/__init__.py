"""
Core application components for the Deepfake Protection API.
"""

from .config import get_settings, get_database_settings, Settings, DatabaseSettings

__all__ = [
    "get_settings",
    "get_database_settings", 
    "Settings",
    "DatabaseSettings",
]