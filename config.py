import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get environment variables
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")

# Validate that the environment variables are set
if not MISTRAL_API_KEY:
    logger.error("MISTRAL_API_KEY environment variable not set.")
    raise ValueError("MISTRAL_API_KEY environment variable not set.")

if not BEARER_TOKEN:
    logger.warning("BEARER_TOKEN environment variable not set. Authentication will fail.")

