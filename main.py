from fastapi import FastAPI, Query, HTTPException, Body, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import comfyuiservice
import requests
import base64
import json
import os
import logging
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="ComfyUI N8n Integration")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict to specific domains as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get webhook URL from environment or use default
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")

# Pydantic models for request validation
class PromptRequest(BaseModel):
    prompt: str
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {"message": "ComfyUI API service is running"}

@app.get("/get-image")
async def get_image(input: str = Query(None)):
    """Legacy endpoint for web interface to get an image."""
    try:
        image = comfyuiservice.fetch_image_from_comfy(input)
        image_stream = io.BytesIO(image)
        return StreamingResponse(image_stream, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.post("/webhook/prompt")
async def receive_prompt(request_data: PromptRequest):
    """
    Receive a prompt from N8n webhook and generate an image.
    Then send the image back to the specified webhook or default N8N_WEBHOOK_URL.
    """
    logger.info(f"Received prompt request: {request_data.prompt[:50]}...")
    
    try:
        # Generate image using ComfyUI
        image_bytes = comfyuiservice.fetch_image_from_comfy(request_data.prompt)
        if not image_bytes:
            raise Exception("No image was generated")
        
        # Determine webhook URL to send the image back to
        webhook_url = request_data.webhook_url or N8N_WEBHOOK_URL
        if not webhook_url:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "error",
                    "message": "No webhook URL provided for sending the image. Set N8N_WEBHOOK_URL environment variable or include webhook_url in the request."
                }
            )
            
        # Encode image as base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare response data
        response_data = {
            "status": "success",
            "image": base64_image,
            "metadata": request_data.metadata
        }
        
        # Send image back to N8n via webhook
        logger.info(f"Sending image to webhook: {webhook_url[:50]}...")
        webhook_response = requests.post(
            webhook_url,
            json=response_data,
            headers={"Content-Type": "application/json"}
        )
        
        if webhook_response.status_code >= 400:
            logger.error(f"Error sending image to webhook: {webhook_response.text}")
            return JSONResponse(
                status_code=207,  # Partial success
                content={
                    "status": "partial_success",
                    "message": "Image generated but webhook delivery failed",
                    "webhook_status": webhook_response.status_code,
                    "webhook_response": webhook_response.text
                }
            )
            
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Image generated and sent to webhook",
                "webhook_status": webhook_response.status_code
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error processing prompt: {str(e)}"
            }
        )

@app.post("/webhook/prompt-direct")
async def receive_prompt_direct(request: Request):
    """
    Alternative endpoint for receiving prompts directly from N8n
    with minimal processing of the request body.
    """
    try:
        # Parse the raw request body
        body = await request.json()
        logger.info(f"Received raw webhook data: {json.dumps(body)[:100]}...")
        
        # Extract prompt from the request body
        # Adjust this based on how N8n formats the data
        prompt = body.get("prompt", "")
        webhook_url = body.get("webhook_url", N8N_WEBHOOK_URL)
        
        if not prompt:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No prompt provided"}
            )
            
        # Generate image using ComfyUI
        image_bytes = comfyuiservice.fetch_image_from_comfy(prompt)
        if not image_bytes:
            raise Exception("No image was generated")
            
        # Proceed with webhook response as in the main endpoint
        if webhook_url:
            # Encode image as base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare response data - include the original body data 
            # so N8n can maintain workflow context
            response_data = {
                "status": "success",
                "image": base64_image,
                "original_data": body
            }
            
            # Send image back to N8n via webhook
            logger.info(f"Sending image to webhook: {webhook_url[:50]}...")
            webhook_response = requests.post(
                webhook_url,
                json=response_data,
                headers={"Content-Type": "application/json"}
            )
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "Image generated and sent to webhook",
                    "webhook_status": webhook_response.status_code
                }
            )
        else:
            # If no webhook URL, return the image directly as base64
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "image": base64.b64encode(image_bytes).decode('utf-8')
                }
            )
            
    except Exception as e:
        logger.error(f"Error in direct webhook processing: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error processing webhook: {str(e)}"}
        )

@app.get("/health")
def health_check():
    """Health check endpoint for N8n to verify service is alive."""
    return {"status": "healthy"}

def run_app():
    """Entry point for the application when installed via pip."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_app()