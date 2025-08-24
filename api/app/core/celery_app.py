"""
Celery configuration for async batch processing.

This module sets up Celery for handling background tasks like
batch image protection processing.
"""

import os
from celery import Celery
from kombu import Queue

from app.core.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "deepfake_protection_api",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.batch_processing",
        "app.tasks.model_management"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "app.tasks.batch_processing.*": {"queue": "batch_processing"},
        "app.tasks.model_management.*": {"queue": "model_management"},
    },
    
    # Queue configuration
    task_default_queue="default",
    task_queues=(
        Queue("default"),
        Queue("batch_processing", routing_key="batch_processing"),
        Queue("model_management", routing_key="model_management"),
    ),
    
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Disable prefetching for memory management
    worker_max_tasks_per_child=10, # Restart workers after 10 tasks
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    
    # Monitoring
    task_send_sent_event=True,
    task_track_started=True,
    
    # Error handling
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Beat schedule (for periodic tasks)
    beat_schedule={
        "cleanup-completed-tasks": {
            "task": "app.tasks.batch_processing.cleanup_completed_tasks",
            "schedule": 3600.0,  # Every hour
        },
        "health-check-models": {
            "task": "app.tasks.model_management.health_check_models",
            "schedule": 300.0,   # Every 5 minutes
        },
    },
)

# Configure logging
celery_app.conf.worker_log_format = "[%(asctime)s: %(levelname)s/%(processName)s] %(message)s"
celery_app.conf.worker_task_log_format = "[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s"


if __name__ == "__main__":
    celery_app.start()