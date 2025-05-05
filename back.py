from PIL import Image
import io
import time
import base64
from diffusers import StableDiffusionPipeline
import torch
import threading
import grpc
from concurrent import futures
import queue
import text_to_image_pb2
import text_to_image_pb2_grpc
import os
import subprocess

# Load model configuration
model_path = "./stable_diffusion_offline"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

# Number of pipelines
MAX_CONCURRENT_REQUESTS = 3  # Number of pipelines and concurrent requests allowed

# Create a queue to manage pipelines
pipeline_queue = queue.Queue()

def create_pipeline():
    """Create a new instance of the StableDiffusionPipeline."""
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe.to(device)
    return pipe

# Initialize pipelines and add them to the queue
for _ in range(MAX_CONCURRENT_REQUESTS):
    pipeline_queue.put(create_pipeline())

class TextToImageServicer(text_to_image_pb2_grpc.TextToImageServicer):
    def GenerateImage(self, request, context):
        prompt = request.prompt
        width = request.width
        height = request.height
        steps = request.steps
        guidance = request.guidance
        dtype = request.dtype  # Get the dtype from the request

        # Validate inputs
        if width < 64 or width > 1024 or height < 64 or height > 1024:
            context.set_details("Width and height must be between 64 and 1024.")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return text_to_image_pb2.ImageResponse(
                status="error",
                message="Invalid dimensions"
            )

        if steps < 1 or steps > 100:
            context.set_details("Steps must be between 1 and 100.")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return text_to_image_pb2.ImageResponse(
                status="error",
                message="Invalid steps"
            )

        if guidance < 1.0 or guidance > 20.0:
            context.set_details("Guidance scale must be between 1.0 and 20.0.")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return text_to_image_pb2.ImageResponse(
                status="error",
                message="Invalid guidance scale"
            )

        if dtype not in ["float32", "float16"]:
            context.set_details("Invalid dtype. Must be 'float32' or 'float16'.")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return text_to_image_pb2.ImageResponse(
                status="error",
                message="Invalid dtype"
            )

        # Determine the torch dtype
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        # Wait for a free pipeline
        try:
            pipe = pipeline_queue.get(block=True, timeout=1)
        except queue.Empty:
            context.set_details("Server busy, please wait...")
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            return text_to_image_pb2.ImageResponse(
                status="error",
                message="Server busy"
            )

        try:
            # Update the pipeline's dtype dynamically
            pipe.to(device, torch_dtype=torch_dtype)

            # Generate the image
            image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, height=height, width=width).images[0]

            # Convert the image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            # Return the response
            return text_to_image_pb2.ImageResponse(
                base64_image=img_base64,
                status="success",
                message="Image generated successfully"
            )
        except Exception as e:
            context.set_details(f"Error generating image: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return text_to_image_pb2.ImageResponse(
                status="error",
                message="Internal server error"
            )
        finally:
            pipeline_queue.put(pipe)

if __name__ == '__main__':

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    text_to_image_pb2_grpc.add_TextToImageServicer_to_server(TextToImageServicer(), server)
    server.add_insecure_port('[::]:50051')  # Port to listen on
    server.start()
    print("Server started, listening on port 50051")

    try:
        while True:
            time.sleep(86400)  # Keep the server running
    except KeyboardInterrupt:
        server.stop(0)
