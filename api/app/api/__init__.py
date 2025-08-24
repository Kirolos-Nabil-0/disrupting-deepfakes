"""
API package for the Deepfake Protection API.
"""

from .v1.router import api_router

__all__ = [
    "api_router",
]