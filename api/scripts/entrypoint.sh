#!/bin/bash
set -e

# Deepfake Protection API Docker Entrypoint Script

echo "Starting Deepfake Protection API..."

# Wait for Redis to be ready
echo "Waiting for Redis..."
until redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping; do
  echo "Redis is unavailable - sleeping"
  sleep 1
done
echo "Redis is ready!"

# Create necessary directories
mkdir -p /app/logs /app/uploads /app/temp /app/models

# Set permissions
chown -R app:app /app/logs /app/uploads /app/temp

# Run database migrations if needed (future feature)
# python manage.py migrate

# Preload models if configured
if [ "${PRELOAD_MODELS:-false}" = "true" ]; then
    echo "Preloading models..."
    python -c "
import asyncio
from app.core.model_manager import ModelManager
from app.core.config import get_settings

async def preload():
    settings = get_settings()
    manager = ModelManager(
        models_dir=settings.models_dir,
        device=settings.device,
        cache_models=True
    )
    try:
        await manager.load_default_models()
        print('Models preloaded successfully')
    except Exception as e:
        print(f'Model preloading failed: {e}')
    finally:
        await manager.cleanup()

asyncio.run(preload())
"
fi

# Check GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
"

# Execute the main command
echo "Starting application with command: $@"
exec "$@"