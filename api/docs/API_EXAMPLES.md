# API Usage Examples

This document provides comprehensive examples for using the Deepfake Protection API.

## Table of Contents

1. [Authentication](#authentication)
2. [Single Image Protection](#single-image-protection)
3. [Batch Processing](#batch-processing)
4. [Model Management](#model-management)
5. [Configuration and Monitoring](#configuration-and-monitoring)
6. [Error Handling](#error-handling)
7. [Client Libraries](#client-libraries)

## Authentication

Currently, the API uses simple rate limiting. Future versions will include API key authentication.

```python
# Headers for future authentication
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

## Single Image Protection

### Basic Protection

```python
import requests
import base64
from pathlib import Path

def protect_image(image_path: str, output_path: str):
    """Protect a single image and save the result."""
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # API request
    response = requests.post(
        "http://localhost:8000/api/v1/protect/image",
        json={
            "image": image_data,
            "model_type": "stargan",
            "attack_method": "pgd",
            "attack_params": {
                "epsilon": 0.05,
                "iterations": 10,
                "step_size": 0.01,
                "random_start": True
            },
            "output_format": "jpeg"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Save protected image
        protected_data = base64.b64decode(result["protected_image"])
        with open(output_path, "wb") as f:
            f.write(protected_data)
        
        # Print metrics
        metrics = result["protection_metrics"]
        print(f"Protection completed:")
        print(f"  L2 norm: {metrics['l2_norm']}")
        print(f"  L∞ norm: {metrics['linf_norm']}")
        print(f"  SSIM: {metrics['ssim']}")
        print(f"  PSNR: {metrics['psnr']}")
        print(f"  Processing time: {result['processing_time_ms']}ms")
        
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage
result = protect_image("input.jpg", "protected.jpg")
```

### Advanced Protection with Custom Parameters

```python
def advanced_protection(image_path: str, target_attributes: list = None):
    """Advanced protection with custom parameters."""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Custom attack parameters
    attack_params = {
        "epsilon": 0.03,      # Smaller perturbations
        "iterations": 20,     # More iterations for better convergence
        "step_size": 0.005,   # Smaller steps
        "random_start": True
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/protect/image",
        json={
            "image": image_data,
            "model_type": "stargan",
            "attack_method": "pgd",
            "attack_params": attack_params,
            "target_attributes": target_attributes or ["Male", "Young"],
            "output_format": "png"  # PNG for lossless quality
        }
    )
    
    return response.json()

# Protect against specific facial attributes
result = advanced_protection(
    "portrait.jpg", 
    target_attributes=["Black_Hair", "Male", "Young"]
)
```

### Comparing Attack Methods

```python
def compare_attack_methods(image_path: str):
    """Compare different attack methods on the same image."""
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    methods = ["fgsm", "pgd", "ifgsm"]
    results = {}
    
    for method in methods:
        response = requests.post(
            "http://localhost:8000/api/v1/protect/image",
            json={
                "image": image_data,
                "model_type": "stargan",
                "attack_method": method,
                "attack_params": {
                    "epsilon": 0.05,
                    "iterations": 10 if method != "fgsm" else 1,
                    "step_size": 0.01
                },
                "output_format": "jpeg"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            results[method] = {
                "metrics": result["protection_metrics"],
                "processing_time": result["processing_time_ms"],
                "protected_image": result["protected_image"]
            }
            
            # Save each result
            protected_data = base64.b64decode(result["protected_image"])
            with open(f"protected_{method}.jpg", "wb") as f:
                f.write(protected_data)
    
    # Compare results
    print("Attack Method Comparison:")
    print("-" * 50)
    for method, data in results.items():
        metrics = data["metrics"]
        print(f"{method.upper()}:")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  L2 norm: {metrics['l2_norm']:.6f}")
        print(f"  Time: {data['processing_time']}ms")
        print()
    
    return results

# Usage
comparison = compare_attack_methods("test_image.jpg")
```

## Batch Processing

### Synchronous Batch Processing

```python
def batch_protect_sync(image_paths: list):
    """Process multiple images synchronously."""
    
    # Encode all images
    images = []
    for path in image_paths:
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
            images.append(image_data)
    
    response = requests.post(
        "http://localhost:8000/api/v1/protect/batch",
        json={
            "images": images,
            "protection_config": {
                "model_type": "stargan",
                "attack_method": "pgd",
                "attack_params": {
                    "epsilon": 0.05,
                    "iterations": 10,
                    "step_size": 0.01
                },
                "output_format": "jpeg"
            },
            "async_processing": False  # Synchronous processing
        }
    )
    
    if response.status_code == 200:
        task_info = response.json()
        task_id = task_info["task_id"]
        
        # Get results immediately (sync processing)
        status_response = requests.get(
            f"http://localhost:8000/api/v1/protect/status/{task_id}"
        )
        
        if status_response.status_code == 200:
            results = status_response.json()
            
            # Save all protected images
            for i, result in enumerate(results["results"]):
                if not result.get("error"):
                    protected_data = base64.b64decode(result["protected_image"])
                    output_path = f"batch_protected_{i}.jpg"
                    with open(output_path, "wb") as f:
                        f.write(protected_data)
                    print(f"Saved: {output_path}")
                else:
                    print(f"Image {i} failed: {result['error']}")
            
            return results
    
    return None

# Usage
image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_results = batch_protect_sync(image_list)
```

### Asynchronous Batch Processing

```python
import time

def batch_protect_async(image_paths: list):
    """Process multiple images asynchronously with progress monitoring."""
    
    # Encode images
    images = []
    for path in image_paths:
        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
            images.append(image_data)
    
    # Start batch processing
    response = requests.post(
        "http://localhost:8000/api/v1/protect/batch",
        json={
            "images": images,
            "protection_config": {
                "model_type": "ganimation",
                "attack_method": "ifgsm",
                "attack_params": {"epsilon": 0.04, "iterations": 15}
            },
            "async_processing": True  # Asynchronous processing
        }
    )
    
    if response.status_code != 200:
        print(f"Failed to start batch: {response.text}")
        return None
    
    task_info = response.json()
    task_id = task_info["task_id"]
    print(f"Batch started with task ID: {task_id}")
    print(f"Estimated completion: {task_info['estimated_completion_ms']}ms")
    
    # Monitor progress
    while True:
        status_response = requests.get(
            f"http://localhost:8000/api/v1/protect/status/{task_id}"
        )
        
        if status_response.status_code == 200:
            status = status_response.json()
            progress = status["progress"]
            
            print(f"Progress: {progress['processed']}/{progress['total']} "
                  f"({progress['percentage']:.1f}%)")
            
            if status["status"] == "completed":
                print("Batch processing completed!")
                
                # Save results
                for i, result in enumerate(status["results"]):
                    if not result.get("error"):
                        protected_data = base64.b64decode(result["protected_image"])
                        output_path = f"async_batch_{i}.jpg"
                        with open(output_path, "wb") as f:
                            f.write(protected_data)
                        print(f"Saved: {output_path}")
                
                return status
            
            elif status["status"] == "failed":
                print(f"Batch processing failed: {status.get('error')}")
                return status
            
            # Wait before next check
            time.sleep(2)
        
        else:
            print(f"Failed to get status: {status_response.text}")
            break
    
    return None

# Usage
large_batch = [f"image_{i}.jpg" for i in range(10)]
async_results = batch_protect_async(large_batch)
```

## Model Management

### Loading and Managing Models

```python
def manage_models():
    """Demonstrate model management operations."""
    
    # List available models
    response = requests.get("http://localhost:8000/api/v1/models")
    if response.status_code == 200:
        models = response.json()["models"]
        print("Available models:")
        for model in models:
            print(f"  {model['name']}: {model['status']} "
                  f"({model.get('memory_usage_mb', 0)}MB)")
    
    # Load a specific model
    response = requests.post("http://localhost:8000/api/v1/models/cyclegan/load")
    if response.status_code == 200:
        print("CycleGAN model loaded successfully")
    else:
        print(f"Failed to load model: {response.text}")
    
    # Unload a model to free memory
    response = requests.post("http://localhost:8000/api/v1/models/pix2pixhd/unload")
    if response.status_code == 200:
        print("pix2pixHD model unloaded")

# Usage
manage_models()
```

### Benchmarking Models

```python
def benchmark_models():
    """Benchmark different models for performance comparison."""
    
    # Test image
    with open("benchmark_image.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    models = ["stargan", "ganimation", "cyclegan"]
    benchmarks = {}
    
    for model in models:
        print(f"Benchmarking {model}...")
        
        # Multiple runs for averaging
        times = []
        for _ in range(5):
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:8000/api/v1/protect/image",
                json={
                    "image": image_data,
                    "model_type": model,
                    "attack_method": "fgsm",  # Fastest method for benchmarking
                    "attack_params": {"epsilon": 0.05}
                }
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                times.append(result["processing_time_ms"])
            else:
                print(f"  Failed: {response.text}")
                break
        
        if times:
            avg_time = sum(times) / len(times)
            benchmarks[model] = {
                "avg_time_ms": avg_time,
                "min_time_ms": min(times),
                "max_time_ms": max(times)
            }
            print(f"  Average time: {avg_time:.1f}ms")
    
    # Print comparison
    print("\nBenchmark Results:")
    print("-" * 40)
    for model, stats in benchmarks.items():
        print(f"{model}: {stats['avg_time_ms']:.1f}ms "
              f"(min: {stats['min_time_ms']:.1f}, max: {stats['max_time_ms']:.1f})")
    
    return benchmarks

# Usage
benchmark_results = benchmark_models()
```

## Configuration and Monitoring

### Health Checks

```python
def check_system_health():
    """Monitor system health and status."""
    
    # Basic health check
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        health = response.json()
        print(f"System status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Models loaded: {health['models_loaded']}")
    
    # Detailed health check
    response = requests.get("http://localhost:8000/api/v1/health")
    if response.status_code == 200:
        detailed_health = response.json()
        print(f"\nDetailed Health Check:")
        print(f"GPU available: {detailed_health['gpu_available']}")
        
        if detailed_health.get('memory_usage'):
            mem = detailed_health['memory_usage']
            print(f"Memory usage:")
            if 'gpu_total_gb' in mem:
                print(f"  GPU: {mem['gpu_allocated_gb']:.1f}GB / {mem['gpu_total_gb']:.1f}GB")
            print(f"  CPU: {mem['cpu_used_gb']:.1f}GB / {mem['cpu_total_gb']:.1f}GB")

# Usage
check_system_health()
```

### Attack Method Information

```python
def get_attack_info():
    """Get information about available attack methods."""
    
    response = requests.get("http://localhost:8000/api/v1/config/attack-methods")
    if response.status_code == 200:
        methods = response.json()["attack_methods"]
        
        for method in methods:
            print(f"\n{method['name']}:")
            print(f"  Description: {method['description']}")
            print(f"  Parameters:")
            
            for param_name, param_info in method['parameters'].items():
                print(f"    {param_name}: {param_info['description']}")
                print(f"      Type: {param_info['type']}")
                print(f"      Default: {param_info['default']}")
                if 'range' in param_info:
                    print(f"      Range: {param_info['range']}")

# Usage
get_attack_info()
```

## Error Handling

### Robust Client Implementation

```python
import requests
from typing import Optional, Dict, Any
import time

class DeepfakeProtectionClient:
    """Robust client with error handling and retries."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def protect_image(
        self, 
        image_path: str, 
        model_type: str = "stargan",
        attack_method: str = "pgd",
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Protect image with error handling and retries."""
        
        try:
            # Read and encode image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"Failed to read image: {e}")
            return None
        
        payload = {
            "image": image_data,
            "model_type": model_type,
            "attack_method": attack_method,
            "attack_params": {
                "epsilon": 0.05,
                "iterations": 10,
                "step_size": 0.01
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/v1/protect/image",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 60))
                    print(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                elif response.status_code == 503:  # Service unavailable
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Service unavailable. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    error = response.json().get("error", {})
                    print(f"API error: {error.get('message', 'Unknown error')}")
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
            except requests.exceptions.ConnectionError:
                print(f"Connection error (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        
        print(f"Failed after {max_retries} attempts")
        return None
    
    def check_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

# Usage
client = DeepfakeProtectionClient()

# Check if service is available
if client.check_health():
    result = client.protect_image("test.jpg")
    if result:
        print("Protection successful!")
    else:
        print("Protection failed!")
else:
    print("Service is not available")
```

## Client Libraries

### Python Client Class

```python
class DeepfakeProtectionAPI:
    """Complete Python client for the Deepfake Protection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })
    
    def protect_single_image(self, **kwargs) -> Dict[str, Any]:
        """Protect a single image."""
        return self._request("POST", "/api/v1/protect/image", json=kwargs)
    
    def start_batch_protection(self, **kwargs) -> Dict[str, Any]:
        """Start batch image protection."""
        return self._request("POST", "/api/v1/protect/batch", json=kwargs)
    
    def get_batch_status(self, task_id: str) -> Dict[str, Any]:
        """Get batch processing status."""
        return self._request("GET", f"/api/v1/protect/status/{task_id}")
    
    def list_models(self) -> Dict[str, Any]:
        """List available models."""
        return self._request("GET", "/api/v1/models")
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model."""
        return self._request("POST", f"/api/v1/models/{model_name}/load")
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a specific model."""
        return self._request("POST", f"/api/v1/models/{model_name}/unload")
    
    def get_attack_methods(self) -> Dict[str, Any]:
        """Get available attack methods."""
        return self._request("GET", "/api/v1/config/attack-methods")
    
    def health_check(self) -> Dict[str, Any]:
        """Get system health status."""
        return self._request("GET", "/api/v1/health")
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API request with error handling."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

# Usage example
api = DeepfakeProtectionAPI("http://localhost:8000")

# Protect an image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

result = api.protect_single_image(
    image=image_data,
    model_type="stargan",
    attack_method="pgd"
)

if "error" not in result:
    print("Protection successful!")
    print(f"Processing time: {result['processing_time_ms']}ms")
else:
    print(f"Error: {result['error']}")
```

### JavaScript/Node.js Example

```javascript
// Node.js client example
const axios = require('axios');
const fs = require('fs');

class DeepfakeProtectionAPI {
    constructor(baseURL = 'http://localhost:8000', apiKey = null) {
        this.client = axios.create({
            baseURL: baseURL,
            timeout: 30000,
            headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}
        });
    }

    async protectImage(imagePath, options = {}) {
        try {
            // Read and encode image
            const imageBuffer = fs.readFileSync(imagePath);
            const imageData = imageBuffer.toString('base64');

            const response = await this.client.post('/api/v1/protect/image', {
                image: imageData,
                model_type: options.modelType || 'stargan',
                attack_method: options.attackMethod || 'pgd',
                attack_params: options.attackParams || {
                    epsilon: 0.05,
                    iterations: 10,
                    step_size: 0.01
                },
                output_format: options.outputFormat || 'jpeg'
            });

            return response.data;
        } catch (error) {
            console.error('Protection failed:', error.response?.data || error.message);
            throw error;
        }
    }

    async healthCheck() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            return { status: 'unhealthy', error: error.message };
        }
    }
}

// Usage
const api = new DeepfakeProtectionAPI();

api.protectImage('input.jpg')
    .then(result => {
        console.log('Protection completed!');
        console.log(`Processing time: ${result.processing_time_ms}ms`);
        
        // Save protected image
        const protectedData = Buffer.from(result.protected_image, 'base64');
        fs.writeFileSync('protected.jpg', protectedData);
    })
    .catch(error => {
        console.error('Failed to protect image:', error);
    });
```

This documentation provides comprehensive examples for using the Deepfake Protection API across different scenarios and programming languages.