import os
import logging
# from dotenv import load_dotenv
# load_dotenv()

# Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- !! IMPORTANT !! ---
# SET YOUR API KEY IN YOUR ENVIRONMENT
# On Windows (Command Prompt): set GEMINI_API_KEY=YOUR_API_KEY_HERE
# On Windows (PowerShell):   $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
# On macOS/Linux:           export GEMINI_API_KEY='YOUR_API_KEY_HERE'
# -------------------------


# HUGGINGFACE CONFIGURATION 
HUGGINGFACETOKEN = os.environ["HUGGINGFACETOKEN"]
# HUGGINGFACETOKEN = os.getenv("HUGGINGFACETOKEN")

# QDRANT CLOUD CONFIGURATION
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_ENDPOINT_URL = os.environ["QDRANT_ENDPOINT_URL"]
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# QDRANT_ENDPOINT_URL = os.getenv("QDRANT_ENDPOINT_URL")

# GEMINI CONFIGURATION
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# WEIGHT & BIASES EVALUATION CONFIGURATION
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
