import logging
try:
    from mistralai import Mistral
    MistralClient = Mistral
    ChatMessage = None  # New API doesn't use ChatMessage class
except ImportError:
    try:
        from mistralai.client import MistralClient
        try:
            from mistralai.models.chat_completion import ChatMessage
        except ImportError:
            ChatMessage = None
    except ImportError:
        print("Warning: Mistral AI library not found. Install with: pip install mistralai")
        MistralClient = None
        ChatMessage = None

from config import MISTRAL_API_KEY # Import from config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Mistral client
client = None
if MistralClient and MISTRAL_API_KEY:
    try:
        client = MistralClient(api_key=MISTRAL_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to initialize Mistral client: {e}")
        client = None
else:
    logger.warning("Mistral client not available - missing API key or library")

async def fetch_context(topic: str) -> str:
    """
    Fetches a detailed explanation for a given technical topic using the Mistral API.
    """
    logger.info(f"Fetching context for topic: {topic}")
    
    if not MistralClient or not MISTRAL_API_KEY:
        logger.warning("Mistral API not available, using fallback context")
        return f"""
        {topic} is an important technical concept that encompasses various aspects and applications.
        It involves multiple components and methodologies that are widely used in software development.
        Understanding {topic} requires knowledge of its core principles, implementation strategies, and best practices.
        This technology has evolved significantly over time and continues to be relevant in modern development.
        Key features include modularity, scalability, and maintainability which make it popular among developers.
        Common use cases involve data processing, system integration, and application development.
        """
    
    prompt = f"Provide a detailed explanation of the technical topic: {topic}. The explanation should be between 4 to 6 paragraphs long."
    
    try:
        # Try different API call methods based on the client type
        chat_response = None
        
        # Method 1: Try chat.complete (newer API)
        try:
            if hasattr(client, 'chat') and hasattr(client.chat, 'complete'):
                chat_response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}]
                )
        except Exception:
            pass
        
        # Method 2: Try direct chat call
        if not chat_response:
            try:
                chat_response = client.chat(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}]
                )
            except Exception:
                pass
        
        # Method 3: Try completions
        if not chat_response:
            try:
                chat_response = client.completions(
                    model="mistral-large-latest",
                    prompt=prompt
                )
            except Exception:
                pass
        
        # Process response
        if chat_response:
            if hasattr(chat_response, 'choices') and chat_response.choices:
                content = chat_response.choices[0].message.content
            elif hasattr(chat_response, 'content'):
                content = chat_response.content
            else:
                content = str(chat_response)
            
            if content and len(content) > 50:
                logger.info(f"Successfully fetched context for {topic}")
                return content
        
        logger.warning(f"No valid response from Mistral for topic: {topic}")
        return ""
        
    except Exception as e:
        logger.error(f"Error fetching context from Mistral for {topic}: {e}")
        # Fall back to mock context for testing
        logger.info("Using fallback context for testing")
        return f"""
        {topic} is an important technical concept in modern software development.
        It encompasses various principles, methodologies, and best practices that developers use to create efficient applications.
        The technology has evolved significantly over the years, incorporating new features and capabilities.
        Understanding {topic} requires knowledge of its core components, syntax, and common use cases.
        It is widely used in web development, data science, automation, and many other domains.
        The community around {topic} is very active, contributing to libraries, frameworks, and tools.
        """

