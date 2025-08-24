"""
Rate limiting middleware for the Deepfake Protection API.

This module implements rate limiting to prevent abuse and ensure
fair usage of the API resources.
"""

import time
import redis
from typing import Dict, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.exceptions import RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis for distributed rate limiting."""
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.redis_client = redis_client
        self.in_memory_store: Dict[str, Dict[str, float]] = {}
        
        # Connect to Redis if not provided
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Rate limiting using Redis backend")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory storage: {e}")
                self.redis_client = None
    
    async def dispatch(self, request: Request, call_next):
        """Process the request with rate limiting."""
        
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        try:
            self._check_rate_limit(client_id, request.url.path)
        except RateLimitError as e:
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                path=request.url.path,
                limit=e.limit,
                window=e.window
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded: {e.limit} requests per {e.window}",
                        "retry_after": e.retry_after
                    }
                },
                headers={"Retry-After": str(e.retry_after)} if e.retry_after else {}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(client_id)
        if remaining is not None:
            response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Window"] = "60"
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        
        # Try to get user ID from authentication (when implemented)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Use IP address as fallback
        client_ip = request.client.host
        
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    def _check_rate_limit(self, client_id: str, path: str) -> None:
        """Check if client has exceeded rate limit."""
        
        current_time = time.time()
        window_start = int(current_time // 60) * 60  # Start of current minute
        
        # Different limits for different endpoints
        limit = self._get_limit_for_path(path)
        window_size = 60  # 1 minute window
        
        if self.redis_client:
            self._check_rate_limit_redis(client_id, window_start, limit, window_size)
        else:
            self._check_rate_limit_memory(client_id, window_start, limit, window_size)
    
    def _get_limit_for_path(self, path: str) -> int:
        """Get rate limit for specific path."""
        
        # Higher limits for batch operations
        if "/protect/batch" in path:
            return max(1, settings.rate_limit_per_minute // 10)  # Much lower for batch
        
        # Lower limits for resource-intensive single image protection
        if "/protect/image" in path:
            return max(10, settings.rate_limit_per_minute // 2)  # Half the normal limit
        
        # Normal limits for other endpoints
        return settings.rate_limit_per_minute
    
    def _check_rate_limit_redis(
        self, 
        client_id: str, 
        window_start: int, 
        limit: int, 
        window_size: int
    ) -> None:
        """Check rate limit using Redis."""
        
        try:
            key = f"rate_limit:{client_id}:{window_start}"
            
            # Increment counter
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window_size)
            results = pipe.execute()
            
            current_count = results[0]
            
            if current_count > limit:
                retry_after = window_size - (time.time() % window_size)
                raise RateLimitError(
                    limit=limit,
                    window="minute",
                    retry_after=int(retry_after)
                )
                
        except redis.RedisError as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to memory-based rate limiting
            self._check_rate_limit_memory(client_id, window_start, limit, window_size)
    
    def _check_rate_limit_memory(
        self, 
        client_id: str, 
        window_start: int, 
        limit: int, 
        window_size: int
    ) -> None:
        """Check rate limit using in-memory storage."""
        
        # Clean old entries
        current_time = time.time()
        self._cleanup_memory_store(current_time)
        
        if client_id not in self.in_memory_store:
            self.in_memory_store[client_id] = {}
        
        client_data = self.in_memory_store[client_id]
        
        # Count requests in current window
        current_count = 0
        for timestamp in list(client_data.keys()):
            if timestamp >= window_start:
                current_count += client_data[timestamp]
        
        if current_count >= limit:
            retry_after = window_size - (current_time % window_size)
            raise RateLimitError(
                limit=limit,
                window="minute",
                retry_after=int(retry_after)
            )
        
        # Record this request
        if window_start not in client_data:
            client_data[window_start] = 0
        client_data[window_start] += 1
    
    def _get_remaining_requests(self, client_id: str) -> Optional[int]:
        """Get remaining requests for client."""
        
        current_time = time.time()
        window_start = int(current_time // 60) * 60
        limit = settings.rate_limit_per_minute
        
        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}:{window_start}"
                current_count = self.redis_client.get(key)
                current_count = int(current_count) if current_count else 0
                return max(0, limit - current_count)
            except redis.RedisError:
                pass
        
        # Fallback to memory
        if client_id in self.in_memory_store:
            client_data = self.in_memory_store[client_id]
            current_count = client_data.get(window_start, 0)
            return max(0, limit - current_count)
        
        return limit
    
    def _cleanup_memory_store(self, current_time: float) -> None:
        """Clean up old entries from memory store."""
        
        cutoff_time = current_time - 120  # Keep last 2 minutes
        
        for client_id in list(self.in_memory_store.keys()):
            client_data = self.in_memory_store[client_id]
            
            # Remove old timestamps
            for timestamp in list(client_data.keys()):
                if timestamp < cutoff_time:
                    del client_data[timestamp]
            
            # Remove client if no data
            if not client_data:
                del self.in_memory_store[client_id]