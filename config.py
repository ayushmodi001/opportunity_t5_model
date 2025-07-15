import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Validate required environment variables
if not MISTRAL_API_KEY:
    print("Warning: MISTRAL_API_KEY environment variable not found")

# Optional: Add other configuration variables here
MAX_CONTEXT_LENGTH = 800
MAX_QUESTIONS_PER_TOPIC = 10
