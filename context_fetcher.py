import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from config import MISTRAL_API_KEY # Import from config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Mistral client
# MISTRAL_API_KEY is now imported from config
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not available from config.")

client = MistralClient(api_key=MISTRAL_API_KEY)

async def fetch_context(topic: str) -> str:
    """
    Fetches a detailed explanation for a given technical topic using the Mistral API.
    """
    logger.info(f"Fetching context for topic: {topic}")
    
    prompt = f"Provide a detailed explanation of the technical topic: {topic}. The explanation should be between 4 to 6 paragraphs long."
    
    try:
        chat_response = client.chat(
            model="mistral-large-latest",
            messages=[ChatMessage(role="user", content=prompt)]
        )
        
        if chat_response.choices:
            context = chat_response.choices[0].message.content
            logger.info(f"Successfully fetched context for {topic}")
            return context
        else:
            logger.warning(f"No response from Mistral for topic: {topic}")
            return ""
            
    except Exception as e:
        logger.error(f"Error fetching context from Mistral for {topic}: {e}")
        raise

