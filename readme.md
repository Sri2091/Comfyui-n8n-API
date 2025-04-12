# ComfyUI N8n Integration

This application creates a bridge between ComfyUI and N8n, allowing you to:
1. Receive prompts from N8n via webhooks
2. Generate images using ComfyUI
3. Send the generated images back to N8n via webhooks

## Getting Started

### Prerequisites

- ComfyUI installed and running (usually on port 8188)
- N8n installed and running (usually on port 5678)
- Python 3.8+ with pip

### Installation

1. Clone this repository or copy all files to a directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Files Explanation

- `main.py` - The main FastAPI server that handles webhooks
- `comfyuiservice.py` - Service that communicates with ComfyUI
- `.env` - Configuration file for your environment settings
- `requirements.txt` - Python dependencies

## Running the Integration

1. Make sure ComfyUI is running
2. Edit the `.env` file with your specific settings (ComfyUI server, N8n webhook URL)
3. Start the server:
   ```bash
   python main.py
   ```

This will start a server on port 8000 that can receive webhooks from N8n.

## Configuring N8n

In your N8n workflow:

1. Add an HTTP Request node:
   - Method: POST
   - URL: `http://your-server-address:8000/webhook/prompt`
   - Body: JSON
   ```json
   {
     "prompt": "Text description for image generation",
     "webhook_url": "http://your-n8n-address:5678/webhook/your-workflow-id"
   }
   ```

2. Add a Webhook node to receive the generated image
   - This will receive a JSON payload with the base64-encoded image

## How It Works

1. N8n sends a prompt to the FastAPI server via webhook
2. The server passes the prompt to ComfyUI using WebSockets
3. ComfyUI generates the image using the workflow defined in workflow.json
4. The image is sent back to N8n via the webhook URL

## Customizing the Workflow

To use your own ComfyUI workflow:

1. Export your workflow from ComfyUI as JSON
2. Save it as `workflow.json` in the same directory
3. Make sure the workflow includes a "SaveImageWebsocket" node

## Troubleshooting

- If no images are generated, check that ComfyUI is running and the workflow is valid
- If the webhook doesn't receive responses, check your network settings
- Check the console logs for detailed error messages

## API Endpoints

- `POST /webhook/prompt` - Main endpoint for N8n to send prompts
- `GET /get-image?input=text` - Direct endpoint to get an image
- `GET /health` - Health check endpoint
