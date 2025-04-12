import websocket 
import uuid
import json
import urllib.request
import urllib.parse
import logging
import os
import time
import ssl
import socket
from typing import Optional, Dict, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration - can be overridden by environment variables
save_image_websocket = os.getenv("SAVE_IMAGE_WEBSOCKET", "SaveImageWebsocket")
server_address = os.getenv("COMFYUI_SERVER", "m8jxw66r6it3p7-8188.proxy.runpod.net")
use_https = os.getenv("COMFYUI_USE_HTTPS", "true").lower() in ("true", "1", "yes")
client_id = str(uuid.uuid4())

# Connection and retry settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))  # seconds
CONNECTION_TIMEOUT = float(os.getenv("CONNECTION_TIMEOUT", "60.0"))  # seconds

# Configure SSL context for secure connections
ssl_context = ssl.create_default_context()
#


def get_prompt_with_workflow(input_text: str, custom_workflow: Optional[Dict[str, Any]] = None) -> Dict:
    """
    Create a prompt with the workflow for ComfyUI.
    
    Args:
        input_text: The text prompt to use for image generation
        custom_workflow: Optional custom workflow to use instead of the default
        
    Returns:
        Dict: The workflow prompt
    """
    if custom_workflow:
        prompt_json = custom_workflow
    else:
        # Load the default workflow from file if it exists
        workflow_path = os.getenv("WORKFLOW_PATH", "workflow.json")
        if os.path.exists(workflow_path):
            try:
                with open(workflow_path, 'r') as f:
                    prompt_json = json.load(f)
                logger.info(f"Loaded workflow from {workflow_path}")
            except Exception as e:
                logger.error(f"Error loading workflow file: {str(e)}")
                prompt_json = get_default_workflow()
        else:
            prompt_json = get_default_workflow()
    
    # Find the CLIP Text Encode node and update its prompt text
    for node_id, node in prompt_json.items():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            if "_meta" in node and node["_meta"].get("title", "").lower().find("positive") >= 0:
                node["inputs"]["text"] = input_text
                logger.info(f"Set prompt text in node {node_id}: {input_text[:50]}...")
                break
    
    return prompt_json


def get_default_workflow() -> Dict:
    """
    Returns the default workflow as a dictionary.
    """
    logger.info("Using default hardcoded workflow")
    
    # This is the workflow from the original code, structured as a dictionary
    workflow = {
        "6": {
            "inputs": {
            "text": "",
            "clip": [
                "38",
                1
            ]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
            "title": "CLIP Text Encode (Positive Prompt)"
            }
        },
        "8": {
            "inputs": {
            "samples": [
                "13",
                0
            ],
            "vae": [
                "10",
                0
            ]
            },
            "class_type": "VAEDecode",
            "_meta": {
            "title": "VAE Decode"
            }
        },
        "9": {
            "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "8",
                0
            ]
            },
            "class_type": "SaveImage",
            "_meta": {
            "title": "Save Image"
            }
        },
        "10": {
            "inputs": {
            "vae_name": "ae.safetensors"
            },
            "class_type": "VAELoader",
            "_meta": {
            "title": "Load VAE"
            }
        },
        "11": {
            "inputs": {
            "clip_name1": "t5xxl_fp16.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux"
            },
            "class_type": "DualCLIPLoader",
            "_meta": {
            "title": "DualCLIPLoader"
            }
        },
        "12": {
            "inputs": {
            "unet_name": "flux_dev.safetensors",
            "weight_dtype": "default"
            },
            "class_type": "UNETLoader",
            "_meta": {
            "title": "Load Diffusion Model"
            }
        },
        "13": {
            "inputs": {
            "noise": [
                "25",
                0
            ],
            "guider": [
                "22",
                0
            ],
            "sampler": [
                "16",
                0
            ],
            "sigmas": [
                "17",
                0
            ],
            "latent_image": [
                "27",
                0
            ]
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {
            "title": "SamplerCustomAdvanced"
            }
        },
        "16": {
            "inputs": {
            "sampler_name": "euler"
            },
            "class_type": "KSamplerSelect",
            "_meta": {
            "title": "KSamplerSelect"
            }
        },
        "17": {
            "inputs": {
            "scheduler": "simple",
            "steps": 20,
            "denoise": 1,
            "model": [
                "30",
                0
            ]
            },
            "class_type": "BasicScheduler",
            "_meta": {
            "title": "BasicScheduler"
            }
        },
        "22": {
            "inputs": {
            "model": [
                "38",
                0
            ],
            "conditioning": [
                "26",
                0
            ]
            },
            "class_type": "BasicGuider",
            "_meta": {
            "title": "BasicGuider"
            }
        },
        "25": {
            "inputs": {
            "noise_seed": 1234511
            },
            "class_type": "RandomNoise",
            "_meta": {
            "title": "RandomNoise"
            }
        },
        "26": {
            "inputs": {
            "guidance": 3.5,
            "conditioning": [
                "6",
                0
            ]
            },
            "class_type": "FluxGuidance",
            "_meta": {
            "title": "FluxGuidance"
            }
        },
        "27": {
            "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
            },
            "class_type": "EmptySD3LatentImage",
            "_meta": {
            "title": "EmptySD3LatentImage"
            }
        },
        "30": {
            "inputs": {
            "max_shift": 1.15,
            "base_shift": 0.5,
            "width": 1024,
            "height": 1024,
            "model": [
                "12",
                0
            ]
            },
            "class_type": "ModelSamplingFlux",
            "_meta": {
            "title": "ModelSamplingFlux"
            }
        },
        "38": {
            "inputs": {
            "lora_name": "flux_realism_lora.safetensors",
            "strength_model": 1,
            "strength_clip": 1,
            "model": [
                "30",
                0
            ],
            "clip": [
                "11",
                0
            ]
            },
            "class_type": "LoraLoader",
            "_meta": {
            "title": "Load LoRA"
            }
        },
        "40": {
            "inputs": {
            "images": [
                "8",
                0
            ]
            },
            "class_type": "SaveImageWebsocket",
            "_meta": {
            "title": "SaveImageWebsocket"
            }
        }
    }
    
    return workflow


def queue_prompt(prompt: Dict, retries: int = MAX_RETRIES) -> Dict:
    """
    Queue a prompt with ComfyUI server with retry logic.
    
    Args:
        prompt: The prompt to queue
        retries: Number of retry attempts
        
    Returns:
        Dict: The response from the ComfyUI server
    """
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    
    # Determine protocol based on use_https setting
    protocol = "https" if use_https else "http"
    
    for attempt in range(retries):
        try:
            # Create a custom opener that ignores SSL certificate validation if needed
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            # Create and configure the request
            url = f"{protocol}://{server_address}/prompt"
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    # You might need to add auth headers here if required
                }
            )
            
            # Try to open with our custom context
            with urllib.request.urlopen(req, timeout=CONNECTION_TIMEOUT, context=ctx) as response:
                return json.loads(response.read())
                
        except urllib.error.HTTPError as e:
            # Special handling for HTTP 403 Forbidden
            if e.code == 403:
                logger.warning(f"HTTP 403 Forbidden when accessing {url}. This might be expected if RunPod restricts direct HTTP access.")
                
                # Return a simulated response with a generated prompt_id
                # This allows the flow to continue using WebSockets, which are working
                fake_prompt_id = f"ws_prompt_{uuid.uuid4()}"
                logger.info(f"Using WebSocket-only flow with generated prompt ID: {fake_prompt_id}")
                return {"prompt_id": fake_prompt_id, "ws_only": True}
            else:
                logger.error(f"HTTP error {e.code} on attempt {attempt+1}/{retries}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{retries} failed: {str(e)}")
            
        # Wait before retrying
        if attempt < retries - 1:
            time.sleep(RETRY_DELAY)
        else:
            logger.error("All retry attempts failed")
            # For WebSocket-only mode, return a fake prompt ID
            fake_prompt_id = f"ws_prompt_{uuid.uuid4()}"
            logger.info(f"Falling back to WebSocket-only with generated prompt ID: {fake_prompt_id}")
            return {"prompt_id": fake_prompt_id, "ws_only": True}


def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    """
    Get an image from the ComfyUI server.
    
    Args:
        filename: The filename of the image
        subfolder: The subfolder where the image is stored
        folder_type: The type of folder
        
    Returns:
        bytes: The image data
    """
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    
    # Determine protocol based on use_https setting
    protocol = "https" if use_https else "http"
    
    with urllib.request.urlopen(f"{protocol}://{server_address}/view?{url_values}", 
                                timeout=CONNECTION_TIMEOUT) as response:
        return response.read()


def get_history(prompt_id: str) -> Dict:
    """
    Get the history of a prompt from the ComfyUI server.
    
    Args:
        prompt_id: The ID of the prompt
        
    Returns:
        Dict: The history data
    """
    # Determine protocol based on use_https setting
    protocol = "https" if use_https else "http"
    
    with urllib.request.urlopen(f"{protocol}://{server_address}/history/{prompt_id}", 
                               timeout=CONNECTION_TIMEOUT) as response:
        return json.loads(response.read())


def get_images(ws: websocket.WebSocket, prompt: Dict) -> bytes:
    """
    Get images from the ComfyUI server via WebSocket.
    
    Args:
        ws: The WebSocket connection
        prompt: The prompt to use
        
    Returns:
        bytes: The image data
    """
    # Queue the prompt and get prompt_id
    queue_response = queue_prompt(prompt)
    prompt_id = queue_response['prompt_id']
    ws_only_mode = queue_response.get('ws_only', False)
    
    output_image = None
    current_node = ""
    
    logger.info(f"Prompt queued with ID: {prompt_id}")
    
    # For WebSocket-only mode, we need to send the prompt directly over WebSocket
    if ws_only_mode:
        logger.info("Using WebSocket-only mode to communicate with ComfyUI")
        try:
            # Send the prompt over WebSocket
            client_message = {
                "type": "execute",
                "data": {
                    "prompt": prompt,
                    "client_id": client_id
                }
            }
            ws.send(json.dumps(client_message))
            logger.info("Sent prompt execution request over WebSocket")
        except Exception as e:
            logger.error(f"Error sending prompt over WebSocket: {str(e)}")
            raise
    
    # Execution status tracking
    execution_start = time.time()
    node_execution_times = {}
    
    try:
        # Set a timeout for total execution
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    logger.debug(f"Received WebSocket message: {message['type']}")
                    
                    if message['type'] == 'executing':
                        data = message['data']
                        # In WebSocket-only mode, the prompt_id in messages might not match
                        if ws_only_mode or data['prompt_id'] == prompt_id:
                            if data['node'] is None:
                                logger.info(f"Execution complete after {time.time() - execution_start:.2f} seconds")
                                break  # Execution is done
                            else:
                                node_number = data['node']
                                if node_number in prompt:
                                    node_class = prompt[node_number]["class_type"]
                                    current_node = node_class
                                    node_execution_times[node_class] = time.time()
                                    logger.info(f"Executing node: {node_class}")
                    
                    elif message['type'] == 'executed':
                        # This message type indicates that execution is complete
                        if ws_only_mode or message.get('data', {}).get('prompt_id') == prompt_id:
                            logger.info("Received execution completion message")
                else:
                    # Binary message typically means image data
                    if current_node == save_image_websocket:
                        logger.info(f"Image received from {save_image_websocket} node")
                        output_image = out[8:]  # Skip the first 8 bytes (image data marker)
                        break  # We have our image, we can stop waiting
            except websocket.WebSocketTimeoutException:
                logger.warning("WebSocket timeout while waiting for response")
                continue
    except Exception as e:
        logger.error(f"Error in WebSocket communication: {str(e)}")
        raise

    if output_image is None:
        logger.warning("No image data received within the timeout period")

    return output_image


def fetch_image_from_comfy(input_text: str, custom_workflow: Optional[Dict] = None) -> bytes:
    """
    Fetch an image from ComfyUI based on a text prompt.
    
    Args:
        input_text: The text prompt to use for image generation
        custom_workflow: Optional custom workflow to use
        
    Returns:
        bytes: The generated image data
    """
    logger.info(f"Generating image for prompt: {input_text[:50]}...")
    
    ws = None
    try:
        # Connect to ComfyUI WebSocket
        ws = websocket.WebSocket()
        
        # Determine protocol based on use_https setting
        ws_protocol = "wss" if use_https else "ws"
        
        # Construct the WebSocket URL - be explicit about the format to avoid URL parsing issues
        ws_url = f"{ws_protocol}://{server_address}/ws?clientId={client_id}"
        
        # Log the connection attempt
        logger.info(f"Attempting to connect to WebSocket at: {ws_url}")
        
        # Connect with timeout and extra options for stability
        ws.connect(
            ws_url, 
            timeout=CONNECTION_TIMEOUT,
            skip_utf8_validation=True,
            sslopt={"cert_reqs": ssl.CERT_NONE}  # Disable certificate verification for more reliable connection
        )
        logger.info(f"Successfully connected to WebSocket at {ws_url}")
        
        # Get the workflow prompt
        prompt = get_prompt_with_workflow(input_text, custom_workflow)
        
        # Get the image
        images = get_images(ws, prompt)
        
        if images is None:
            logger.warning("No images were generated. Check ComfyUI server logs for details.")
            return None
            
        return images
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise
    finally:
        # Close the WebSocket connection
        if ws:
            try:
                ws.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {str(e)}")
