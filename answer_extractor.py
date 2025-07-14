import spacy
import yake
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Custom stop words to filter out from answers
CUSTOM_STOP_WORDS = {
    "new", "etc", "use", "application", "applications", "way", "ways", "use case", "use cases",
    "thing", "things", "some", "any", "lot", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "number", "numbers", "example", "examples", "a", "an", "the",
    "of", "in", "on", "at", "for", "to", "with", "by", "from", "and", "or", "but", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "another practical application", "cleaner, more maintainable, and more modular code",
    "a cleaner and more modular codebase"
}

def extract_noun_chunks(text: str) -> list:
    """Extracts noun chunks from text using spaCy."""
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

async def extract_answers(context: str) -> list:
    """
    Extracts up to 10 key answer terms from the context.
    It combines YAKE for keyword extraction and spaCy for noun chunking,
    then filters for uniqueness and relevance.
    """
    logger.info("Extracting answers from context...")
    
    # 1. Use YAKE to get initial keywords
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=30, features=None)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(context)]
    
    # 2. Use spaCy to get noun chunks
    noun_chunks = extract_noun_chunks(context)
    
    # 3. Combine keywords and noun chunks
    combined_keywords = list(set(keywords) | set(noun_chunks))

    # 4. Filter out keywords that are substrings of other keywords or are stop words
    unique_keywords = []
    # Sort by length descending to ensure we keep the longest variants
    for kw in sorted(combined_keywords, key=len, reverse=True):
        kw_lower = kw.lower()
        if kw_lower in CUSTOM_STOP_WORDS:
            continue
        if not any(kw_lower in other_kw.lower() for other_kw in unique_keywords):
            unique_keywords.append(kw)

    # 5. Limit to top 10 diverse keywords
    final_answers = unique_keywords[:10]
    
    if not final_answers:
        logger.warning("Could not extract any answers.")
        return []
        
    logger.info(f"Extracted and filtered answers: {final_answers}")
    return final_answers
