# Deepfake Protection API

A FastAPI-based web service that provides deepfake protection capabilities by applying adversarial perturbations to input images. The API serves as a wrapper around existing deepfake protection models (StarGAN, GANimation, pix2pixHD, CycleGAN) and their adversarial attack implementations.

## Features

- 🛡️ **Image Protection**: Apply adversarial perturbations to protect images from deepfake manipulation
- 🤖 **Multiple Models**: Support for StarGAN, GANimation, pix2pixHD, and CycleGAN models  
- ⚡ **Multiple Attack Methods**: FGSM, PGD, and I-FGSM attack implementations
- 📊 **Quality Metrics**: Calculate protection quality metrics (L2/L∞ norm, SSIM, PSNR)
- 🔄 **Batch Processing**: Process multiple images with async capabilities
- 📈 **Monitoring**: Health checks, metrics, and comprehensive logging
- 🐳 **Docker Support**: Full containerization with GPU support
- 🚀 **Production Ready**: Rate limiting, error handling, and security features

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- Docker and Docker Compose (for containerized deployment)
- Redis (for caching and task queue)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd disrupting-deepfakes/api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start Redis**
```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or install locally
sudo apt-get install redis-server
redis-server
```

5. **Run the API**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Docker Deployment

### Development

```bash
# Start all services
docker-compose up --build

# API: http://localhost:8000
# Flower (Celery monitoring): http://localhost:5555
```

### Production

```bash
# Start production services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With reverse proxy
docker-compose --profile production up -d
```

## API Usage

### Single Image Protection

```python
import requests
import base64

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Protect image
response = requests.post("http://localhost:8000/api/v1/protect/image", json={
    "image": image_data,
    "model_type": "stargan",
    "attack_method": "pgd",
    "attack_params": {
        "epsilon": 0.05,
        "iterations": 10,
        "step_size": 0.01
    },
    "output_format": "jpeg"
})

result = response.json()
protected_image = result["protected_image"]
```

### Batch Processing

```python
# Start batch processing
response = requests.post("http://localhost:8000/api/v1/protect/batch", json={
    "images": [image_data1, image_data2],
    "protection_config": {
        "model_type": "stargan",
        "attack_method": "pgd",
        "attack_params": {"epsilon": 0.05}
    },
    "async_processing": True
})

task_id = response.json()["task_id"]

# Check status
status_response = requests.get(f"http://localhost:8000/api/v1/protect/status/{task_id}")
```

## API Endpoints

### Protection

- `POST /api/v1/protect/image` - Protect single image
- `POST /api/v1/protect/batch` - Start batch protection
- `GET /api/v1/protect/status/{task_id}` - Get batch status

### Model Management

- `GET /api/v1/models` - List available models
- `POST /api/v1/models/{model_name}/load` - Load model
- `POST /api/v1/models/{model_name}/unload` - Unload model

### Configuration

- `GET /api/v1/config/attack-methods` - List attack methods
- `GET /api/v1/health` - Health check

### Documentation

- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Configuration

### Environment Variables

```bash
# API Settings
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-secret-key
ALLOWED_ORIGINS=http://localhost:3000
RATE_LIMIT_PER_MINUTE=60

# Models
MODELS_DIR=./models
PRELOAD_MODELS=true
DEVICE=auto  # auto, cpu, cuda

# Image Processing
MAX_IMAGE_SIZE_MB=10
MAX_IMAGE_DIMENSION=2048
SUPPORTED_FORMATS=jpeg,png,webp

# Redis
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379/0

# Batch Processing
MAX_BATCH_SIZE=10
BATCH_TIMEOUT_SECONDS=300
```

### Model Configuration

Models should be placed in the `models/` directory with the following structure:

```
models/
├── stargan/
│   ├── 200000-G.ckpt
│   └── config.json
├── ganimation/
│   ├── model.pth
│   └── config.json
├── pix2pixhd/
│   ├── latest_net_G.pth
│   └── config.json
└── cyclegan/
    ├── latest_net_G_A.pth
    └── config.json
```

## Attack Methods

### Fast Gradient Sign Method (FGSM)
- Single-step gradient-based attack
- Parameters: `epsilon`

### Projected Gradient Descent (PGD)
- Multi-step iterative attack with projection
- Parameters: `epsilon`, `iterations`, `step_size`, `random_start`

### Iterative FGSM (I-FGSM)
- Multi-step FGSM variant
- Parameters: `epsilon`, `iterations`, `step_size`

## Monitoring and Logging

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed service health
curl http://localhost:8000/api/v1/health
```

### Metrics

Prometheus metrics are available at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

### Logs

Structured logging with JSON format:

```bash
# View logs
docker-compose logs -f api

# Filter by component
docker-compose logs -f api | grep "model_manager"
```

## Performance Optimization

### GPU Utilization

- Automatic GPU detection and utilization
- Memory management and cleanup
- Batch processing for improved throughput

### Caching

- Model caching to reduce loading times
- Redis-based result caching
- Memory-efficient image processing

### Rate Limiting

- Per-IP and per-user rate limiting
- Different limits for different endpoints
- Configurable burst protection

## Development

### Project Structure

```
api/
├── app/
│   ├── api/v1/          # API endpoints
│   ├── core/            # Core components
│   ├── models/          # Data models
│   ├── services/        # Business logic
│   ├── middleware/      # Custom middleware
│   └── tasks/           # Celery tasks
├── models/              # Model files
├── tests/               # Test suite
├── scripts/             # Utility scripts
└── docs/                # Documentation
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# All tests with coverage
pytest --cov=app tests/
```

### Adding New Models

1. Implement model loading in `app/core/model_manager.py`
2. Add model type to `app/models/schemas.py`
3. Update attack engine if needed
4. Add tests for new model

### Adding New Attack Methods

1. Implement attack class in `app/core/attacks.py`
2. Add to `AttackFactory.create_attack()`
3. Update method information in `get_available_methods()`
4. Add comprehensive tests

## Security Considerations

- Input validation for all endpoints
- Rate limiting to prevent abuse
- File size and format restrictions
- Error handling without information leakage
- CORS configuration for web clients

## Troubleshooting

### Common Issues

**GPU Out of Memory**
```bash
# Reduce batch size or model cache
export MAX_BATCH_SIZE=2
export CACHE_MODELS=false
```

**Model Loading Fails**
```bash
# Check model files exist
ls -la models/

# Verify model paths in logs
docker-compose logs api | grep "model"
```

**Redis Connection Issues**
```bash
# Check Redis status
redis-cli ping

# Restart Redis
docker-compose restart redis
```

### Performance Tuning

**For High Throughput**
```bash
# Increase worker processes
export WORKERS=4

# Use multiple Celery workers
docker-compose up --scale worker=3
```

**For Low Memory**
```bash
# Disable model preloading
export PRELOAD_MODELS=false

# Reduce image size limits
export MAX_IMAGE_SIZE_MB=5
export MAX_IMAGE_DIMENSION=1024
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation at `/docs`
- Review the health check endpoint for system status