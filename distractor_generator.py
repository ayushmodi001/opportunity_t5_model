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
        print("Warning: Mistral AI library not found")
        MistralClient = None
        ChatMessage = None

import random
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
    logger.warning("MISTRAL_API_KEY not set. Distractor generation will use dummy fallbacks.")

async def generate_distractors(context: str, correct_answer: str) -> list:
    """
    Generates 3 plausible incorrect options (distractors) for a given answer.
    Uses Mistral API and falls back to dummy distractors on failure.
    """
    if not client:
        return generate_dummy_distractors(correct_answer)

    logger.info(f"Generating distractors for answer: {correct_answer}")
    
    prompt = f"""
    Given the context below and the correct answer, generate 3 plausible but incorrect options (distractors).
    The distractors should be in the same style and format as the correct answer.
    Return only a Python list of 3 strings.

    Context: "{context}"
    Correct Answer: "{correct_answer}"

    Distractors:
    """
    
    try:
        # Try different API call methods based on the client type
        chat_response = None
        
        # Method 1: Try chat.complete (newer API)
        try:
            if hasattr(client, 'chat') and hasattr(client.chat, 'complete'):
                chat_response = client.chat.complete(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
        except Exception:
            pass
        
        # Method 2: Try direct chat call
        if not chat_response:
            try:
                chat_response = client.chat(
                    model="mistral-small-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
            except Exception:
                pass
        
        # Process response
        if chat_response:
            if hasattr(chat_response, 'choices') and chat_response.choices:
                response_text = chat_response.choices[0].message.content.strip()
                
                # Attempt to parse the list from the response
                try:
                    # The model might return a string representation of a list
                    distractors = eval(response_text)
                    if isinstance(distractors, list) and len(distractors) == 3:
                        logger.info(f"Successfully generated distractors for '{correct_answer}'")
                        return distractors
                except:
                    pass # Fallback if eval fails

    except Exception as e:
        logger.error(f"Error generating distractors with Mistral for '{correct_answer}': {e}")

    # Fallback to dummy distractors
    logger.warning(f"Falling back to dummy distractors for '{correct_answer}'")
    return generate_dummy_distractors(correct_answer)


def generate_dummy_distractors(correct_answer: str) -> list:
    """Generates simple, generic dummy distractors."""
    # Basic variations
    dummies = [
        f"Not {correct_answer}",
        f"Alternative to {correct_answer}",
        f"Opposite of {correct_answer}",
        "None of the above",
        "All of the above"
    ]
    
    # Shuffle and pick 3
    random.shuffle(dummies)
    return dummies[:3]
