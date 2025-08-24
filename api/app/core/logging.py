"""
Structured logging configuration for the Deepfake Protection API.

This module sets up structured logging using structlog with support for
both JSON and text formats, request tracing, and performance monitoring.
"""

import logging
import logging.config
import sys
import time
import uuid
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Setup structured logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
    """
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (defaults to caller's module)
    
    Returns:
        Configured structlog BoundLogger
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Logger for HTTP requests with timing and context."""
    
    def __init__(self):
        self.logger = get_logger("request")
    
    def log_request_start(
        self, 
        method: str, 
        path: str, 
        request_id: str,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> float:
        """
        Log the start of a request.
        
        Returns:
            Start timestamp for timing calculations
        """
        start_time = time.time()
        
        self.logger.info(
            "Request started",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent,
            timestamp=start_time
        )
        
        return start_time
    
    def log_request_end(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        start_time: float,
        response_size: Optional[int] = None,
        error: Optional[str] = None
    ) -> None:
        """Log the end of a request with timing information."""
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        log_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "timestamp": end_time
        }
        
        if response_size:
            log_data["response_size"] = response_size
        
        if error:
            log_data["error"] = error
            self.logger.error("Request completed with error", **log_data)
        else:
            self.logger.info("Request completed", **log_data)


class ModelLogger:
    """Logger for model operations and performance."""
    
    def __init__(self):
        self.logger = get_logger("model")
    
    def log_model_load_start(self, model_type: str, device: str) -> float:
        """Log the start of model loading."""
        start_time = time.time()
        
        self.logger.info(
            "Model loading started",
            model_type=model_type,
            device=device,
            timestamp=start_time
        )
        
        return start_time
    
    def log_model_load_end(
        self,
        model_type: str,
        device: str,
        start_time: float,
        success: bool,
        memory_usage_mb: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Log the end of model loading."""
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        log_data = {
            "model_type": model_type,
            "device": device,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "timestamp": end_time
        }
        
        if memory_usage_mb:
            log_data["memory_usage_mb"] = round(memory_usage_mb, 2)
        
        if error:
            log_data["error"] = error
            self.logger.error("Model loading failed", **log_data)
        else:
            self.logger.info("Model loading completed", **log_data)
    
    def log_inference(
        self,
        model_type: str,
        attack_method: str,
        image_size: tuple,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log model inference operation."""
        log_data = {
            "model_type": model_type,
            "attack_method": attack_method,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "duration_ms": round(duration_ms, 2),
            "success": success
        }
        
        if error:
            log_data["error"] = error
            self.logger.error("Model inference failed", **log_data)
        else:
            self.logger.info("Model inference completed", **log_data)


class AttackLogger:
    """Logger for adversarial attack operations."""
    
    def __init__(self):
        self.logger = get_logger("attack")
    
    def log_attack_start(
        self,
        attack_method: str,
        model_type: str,
        attack_params: Dict[str, Any]
    ) -> float:
        """Log the start of attack generation."""
        start_time = time.time()
        
        self.logger.info(
            "Attack generation started",
            attack_method=attack_method,
            model_type=model_type,
            attack_params=attack_params,
            timestamp=start_time
        )
        
        return start_time
    
    def log_attack_end(
        self,
        attack_method: str,
        model_type: str,
        start_time: float,
        success: bool,
        perturbation_norm: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Log the end of attack generation."""
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        log_data = {
            "attack_method": attack_method,
            "model_type": model_type,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "timestamp": end_time
        }
        
        if perturbation_norm:
            log_data["perturbation_norm"] = round(perturbation_norm, 6)
        
        if error:
            log_data["error"] = error
            self.logger.error("Attack generation failed", **log_data)
        else:
            self.logger.info("Attack generation completed", **log_data)


class BatchLogger:
    """Logger for batch processing operations."""
    
    def __init__(self):
        self.logger = get_logger("batch")
    
    def log_batch_start(
        self,
        task_id: str,
        batch_size: int,
        model_type: str,
        attack_method: str
    ) -> float:
        """Log the start of batch processing."""
        start_time = time.time()
        
        self.logger.info(
            "Batch processing started",
            task_id=task_id,
            batch_size=batch_size,
            model_type=model_type,
            attack_method=attack_method,
            timestamp=start_time
        )
        
        return start_time
    
    def log_batch_progress(
        self,
        task_id: str,
        processed: int,
        total: int,
        current_duration_ms: float
    ) -> None:
        """Log batch processing progress."""
        percentage = (processed / total) * 100 if total > 0 else 0
        
        self.logger.info(
            "Batch processing progress",
            task_id=task_id,
            processed=processed,
            total=total,
            percentage=round(percentage, 1),
            duration_ms=round(current_duration_ms, 2)
        )
    
    def log_batch_end(
        self,
        task_id: str,
        start_time: float,
        successful: int,
        failed: int,
        total: int,
        error: Optional[str] = None
    ) -> None:
        """Log the end of batch processing."""
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        log_data = {
            "task_id": task_id,
            "successful": successful,
            "failed": failed,
            "total": total,
            "duration_ms": round(duration_ms, 2),
            "timestamp": end_time
        }
        
        if error:
            log_data["error"] = error
            self.logger.error("Batch processing failed", **log_data)
        else:
            self.logger.info("Batch processing completed", **log_data)


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_memory_usage(
        self,
        component: str,
        cpu_memory_mb: float,
        gpu_memory_mb: Optional[float] = None
    ) -> None:
        """Log memory usage statistics."""
        log_data = {
            "component": component,
            "cpu_memory_mb": round(cpu_memory_mb, 2)
        }
        
        if gpu_memory_mb:
            log_data["gpu_memory_mb"] = round(gpu_memory_mb, 2)
        
        self.logger.info("Memory usage", **log_data)
    
    def log_gpu_utilization(
        self,
        gpu_id: int,
        utilization_percent: float,
        memory_used_mb: float,
        memory_total_mb: float
    ) -> None:
        """Log GPU utilization statistics."""
        self.logger.info(
            "GPU utilization",
            gpu_id=gpu_id,
            utilization_percent=round(utilization_percent, 1),
            memory_used_mb=round(memory_used_mb, 2),
            memory_total_mb=round(memory_total_mb, 2),
            memory_usage_percent=round((memory_used_mb / memory_total_mb) * 100, 1)
        )


# Global logger instances
request_logger = RequestLogger()
model_logger = ModelLogger()
attack_logger = AttackLogger()
batch_logger = BatchLogger()
performance_logger = PerformanceLogger()


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def log_exception(logger: structlog.BoundLogger, exception: Exception, **kwargs) -> None:
    """
    Log an exception with full context.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        **kwargs: Additional context to include
    """
    logger.error(
        "Exception occurred",
        exception_type=exception.__class__.__name__,
        exception_message=str(exception),
        **kwargs,
        exc_info=True
    )