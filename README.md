# Da Vinchi - AI-Powered Text-to-Image Generator

## Project Overview
Da Vinchi is a microservice-based application that generates images from text prompts using Stable Diffusion. It includes a gRPC backend for processing requests and a Streamlit-based frontend for user interaction. Users can generate images, apply filters, and save or download their creations.

## Features
- Generate images from text prompts using Stable Diffusion.
- Adjustable parameters:
  - Image dimensions (multiples of 8, up to 1280x720).
  - Inference steps.
  - Guidance scale.
  - Precision type (`float32` or `float16`).
- Image editing tools:
  - Filters (Sketch, Oil Painting, Warm Tone, Cool Tone, Glitch).
  - Brightness, contrast, and saturation adjustments.
- Save and download generated or edited images.
- Timeline to view saved images.

## Architecture
**Backend**:
- **gRPC API**:
  - Endpoint: `/GenerateImage`
  - Handles requests for image generation.
  - Processes requests concurrently using a thread pool.
- **Stable Diffusion**:
  - Model: Pretrained Stable Diffusion model.
  - Framework: Hugging Face Diffusers.

**Frontend**:
- **Streamlit**:
  - User interface for sending requests to the backend.
  - Displays generated images and provides editing tools.

**Communication**:
- The frontend communicates with the backend over gRPC on port `50051`.

## Setup Instructions

### Prerequisites
- Docker Desktop installed.
- Python 3.10+ (if running locally).
- GPU with CUDA support (optional but recommended).

### Steps to Run

**Option 1: Using Docker**
1. Build the Docker image:
   ```bash
   docker build -t da-vinchi-backend .
   ```
2. Run the backend container:
   ```bash
   docker run -d -p 50051:50051 --name da-vinchi-backend da-vinchi-backend
   ```
3. Run the frontend locally:
   ```bash
   streamlit run app.py
   ```

**Option 2: Running Locally**
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the backend:
   ```bash
   python back.py
   ```
3. Start the frontend:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the frontend in your browser (default: `http://localhost:50051`).
2. Enter a text prompt and adjust parameters (e.g., dimensions, steps, guidance scale).
3. Click "Generate Image" to create an image.
4. View the generated image and apply filters or adjustments.
5. Save or download the image.

## Model Sources
- **Stable Diffusion**:
  - Source: Hugging Face Diffusers.
  - Model Path: `stable_diffusion_offline`.

## Limitations
- **Memory Usage**: The Stable Diffusion model requires significant memory. Ensure sufficient resources are allocated.
- **Concurrency**: Limited by the number of pipelines (`MAX_CONCURRENT_REQUESTS`).
- **Resolution**: Maximum resolution is 1280x720 (720p).


### Functional Test Cases
| **Test Case**                | **Input**                                                                 | **Expected Output**                                                                 |
|-------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Valid Request                | Prompt: "A sunset", Width: 512, Height: 512, Steps: 50, Guidance: 7.5     | Image generated successfully.                                                      |
| Invalid Dimensions           | Width: 513, Height: 513                                                  | Error: "Width and height must be multiples of 8."                                  |
| Invalid Steps                | Steps: 150                                                               | Error: "Steps must be between 1 and 100."                                          |
| Invalid Guidance Scale       | Guidance: 25.0                                                           | Error: "Guidance scale must be between 1.0 and 20.0."                              |
| Invalid Precision (dtype)    | Dtype: "float64"                                                         | Error: "Invalid dtype. Must be 'float32' or 'float16'."                            |
| Missing Prompt               | Prompt: ""                                                               | Error: "Prompt cannot be empty."                                                   |
| Server Busy                  | Simulate multiple concurrent requests exceeding `MAX_CONCURRENT_REQUESTS` | Error: "Server busy."                                                              |


#### Metrics
- **Response Time**: Time taken to generate an image.
- **Concurrent Requests**: Number of requests the backend can handle simultaneously.
- **Resource Usage**: CPU and memory usage during high load.

#### Tools
- Use `locust` or `wrk` for load testing.
- Monitor resource usage using Docker Desktop or system tools.

#### Test Results
| **Metric**                  | **Value**                                                                 |
|------------------------------|---------------------------------------------------------------------------|
| Average Response Time        | ~5 seconds (512x512, 50 steps, float16).                                 |
| Maximum Concurrent Requests  | 3 (with `MAX_CONCURRENT_REQUESTS = 3`).                                  |
| Memory Usage                 | ~2.5 GB per pipeline (512x512, float16).                                 |
| CPU Usage                    | ~100% during image generation.                                           |

#### Performance Graphs
1. **Response Time vs. Concurrent Requests**:
   - Plot the response time as the number of concurrent requests increases.
   - Example:
     ```
     Concurrent Requests: 1, Response Time: 5s
     Concurrent Requests: 2, Response Time: 6s
     Concurrent Requests: 3, Response Time: 8s
     ```

2. **Memory Usage Over Time**:
   - Monitor memory usage during high load.
   - Example:
     ```
     Time: 0s, Memory: 1.5 GB
     Time: 10s, Memory: 2.5 GB
     ```

### Recommendations
- **Optimize Backend**:
  - Reduce `MAX_CONCURRENT_REQUESTS` if memory is limited.
  - Use `float16` for lower memory usage.
- **Scale Horizontally**:
  - Deploy multiple backend instances with a load balancer.
- **Use GPU**:
  - Enable GPU acceleration for faster image generation.

### 👥 Contributors
Name	Roll Number
Shaimaan Qadir	i22-0511
Abdul Moiz Rana	i22-0539
Abdul Ahad	i22-0568
