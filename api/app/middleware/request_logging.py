"""
Request logging middleware for the Deepfake Protection API.

This middleware logs all HTTP requests with timing, context, and error information.
"""

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.logging import request_logger, get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests with timing and context."""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        """Process request with logging."""
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract request information
        method = request.method
        path = request.url.path
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")
        content_length = request.headers.get("content-length")
        
        # Log request start
        start_time = request_logger.log_request_start(
            method=method,
            path=path,
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Add request size if available
        request_size = None
        if content_length:
            try:
                request_size = int(content_length)
            except ValueError:
                pass
        
        error_message = None
        status_code = 500  # Default error status
        response_size = None
        
        try:
            # Process request
            response: Response = await call_next(request)
            status_code = response.status_code
            
            # Get response size if available
            if hasattr(response, 'body') and response.body:
                response_size = len(response.body)
            elif "content-length" in response.headers:
                try:
                    response_size = int(response.headers["content-length"])
                except ValueError:
                    pass
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            error_message = str(e)
            status_code = 500
            
            # Log the exception
            logger.error(
                "Request processing failed",
                request_id=request_id,
                method=method,
                path=path,
                error=error_message,
                exc_info=True
            )
            
            # Re-raise the exception
            raise
            
        finally:
            # Log request completion
            request_logger.log_request_end(
                request_id=request_id,
                method=method,
                path=path,
                status_code=status_code,
                start_time=start_time,
                response_size=response_size,
                error=error_message
            )
            
            # Log additional context for specific endpoints
            self._log_endpoint_context(request, status_code, start_time)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client host
        return request.client.host if request.client else "unknown"
    
    def _log_endpoint_context(self, request: Request, status_code: int, start_time: float) -> None:
        """Log additional context for specific endpoints."""
        
        path = request.url.path
        duration_ms = (time.time() - start_time) * 1000
        
        # Log context for protection endpoints
        if "/protect/image" in path:
            logger.info(
                "Image protection request",
                path=path,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                request_id=getattr(request.state, "request_id", None)
            )
        
        elif "/protect/batch" in path:
            logger.info(
                "Batch protection request",
                path=path,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                request_id=getattr(request.state, "request_id", None)
            )
        
        elif "/models" in path and request.method == "POST":
            logger.info(
                "Model management request",
                path=path,
                method=request.method,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                request_id=getattr(request.state, "request_id", None)
            )
        
        # Log slow requests
        if duration_ms > 5000:  # Requests taking more than 5 seconds
            logger.warning(
                "Slow request detected",
                path=path,
                method=request.method,
                duration_ms=round(duration_ms, 2),
                status_code=status_code,
                request_id=getattr(request.state, "request_id", None)
            )
        
        # Log error responses
        if status_code >= 400:
            level = "error" if status_code >= 500 else "warning"
            getattr(logger, level)(
                f"Request failed with status {status_code}",
                path=path,
                method=request.method,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                request_id=getattr(request.state, "request_id", None)
            )