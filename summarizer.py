import os
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mistral Summarizer ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
mistral_client = None
if MISTRAL_API_KEY:
    mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

async def summarize_with_mistral(text: str) -> str:
    """Summarizes text using the Mistral API."""
    if not mistral_client:
        logger.warning("Mistral API key not found. Skipping Mistral summarization.")
        return text

    logger.info("Summarizing with Mistral...")
    prompt = f"Summarize the following text in a concise manner, keeping the essential information:\n\n{text}"
    
    try:
        chat_response = mistral_client.chat(
            model="mistral-small-latest",
            messages=[ChatMessage(role="user", content=prompt)]
        )
        if chat_response.choices:
            return chat_response.choices[0].message.content
        return text
    except Exception as e:
        logger.error(f"Error during Mistral summarization: {e}")
        return text # Fallback to original text

# --- T5 Summarizer ---
T5_MODEL_NAME = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME)

async def summarize_with_t5(text: str) -> str:
    """Summarizes text using the T5 model."""
    logger.info("Summarizing with T5...")
    try:
        # Prepending "summarize: " is important for T5
        inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logger.error(f"Error during T5 summarization: {e}")
        return text # Fallback to original text

async def summarize_text(text: str, use_mistral: bool = True) -> str:
    """
    Summarizes text if it's over 800 words.
    Prefers Mistral if available, otherwise falls back to T5.
    """
    if len(text.split()) <= 800:
        return text

    logger.info("Text exceeds 800 words, summarization required.")
    
    if use_mistral and mistral_client:
        return await summarize_with_mistral(text)
    else:
        return await summarize_with_t5(text)
