import logging
import re
from typing import List
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model for sentence segmentation
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def chunk_context_by_sentences(context: str, target_chunks: int = 10) -> List[str]:
    """
    Break context into meaningful chunks based on sentences.
    Aims to create approximately 'target_chunks' number of chunks.
    """
    # Use spaCy for better sentence segmentation
    doc = nlp(context)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    if len(sentences) <= target_chunks:
        # If we have fewer sentences than target chunks, return each sentence as a chunk
        return sentences
    
    # Calculate sentences per chunk
    sentences_per_chunk = max(1, len(sentences) // target_chunks)
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk = " ".join(chunk_sentences)
        if chunk.strip():
            chunks.append(chunk.strip())
    
    # If we have more chunks than target, combine the smallest ones
    while len(chunks) > target_chunks and len(chunks) > 1:
        # Find the shortest chunk and combine it with its neighbor
        shortest_idx = min(range(len(chunks)), key=lambda i: len(chunks[i]))
        
        if shortest_idx == 0:
            # Combine with next chunk
            chunks[0] = chunks[0] + " " + chunks[1]
            chunks.pop(1)
        elif shortest_idx == len(chunks) - 1:
            # Combine with previous chunk
            chunks[shortest_idx - 1] = chunks[shortest_idx - 1] + " " + chunks[shortest_idx]
            chunks.pop(shortest_idx)
        else:
            # Combine with shorter neighbor
            if len(chunks[shortest_idx - 1]) <= len(chunks[shortest_idx + 1]):
                chunks[shortest_idx - 1] = chunks[shortest_idx - 1] + " " + chunks[shortest_idx]
                chunks.pop(shortest_idx)
            else:
                chunks[shortest_idx] = chunks[shortest_idx] + " " + chunks[shortest_idx + 1]
                chunks.pop(shortest_idx + 1)
    
    logger.info(f"Created {len(chunks)} context chunks from {len(sentences)} sentences")
    return chunks

def chunk_context_by_paragraphs(context: str, target_chunks: int = 10) -> List[str]:
    """
    Break context into chunks based on paragraphs and logical breaks.
    """
    # Split by double newlines (paragraph breaks)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', context) if p.strip()]
    
    if len(paragraphs) >= target_chunks:
        # If we have enough paragraphs, return them as is (up to target)
        return paragraphs[:target_chunks]
    
    # If we have fewer paragraphs, break them down further by sentences
    all_chunks = []
    chunks_per_paragraph = max(1, target_chunks // len(paragraphs))
    
    for paragraph in paragraphs:
        para_chunks = chunk_context_by_sentences(paragraph, chunks_per_paragraph)
        all_chunks.extend(para_chunks)
    
    # Limit to target number of chunks
    return all_chunks[:target_chunks]

def chunk_context_smart(context: str, target_chunks: int = 10) -> List[str]:
    """
    Intelligently chunk context using a combination of paragraph and sentence-based approaches.
    """
    logger.info(f"Chunking context into approximately {target_chunks} pieces")
    
    # First try paragraph-based chunking
    chunks = chunk_context_by_paragraphs(context, target_chunks)
    
    # If we don't have enough chunks, fall back to sentence-based
    if len(chunks) < target_chunks // 2:
        logger.info("Not enough paragraph-based chunks, using sentence-based chunking")
        chunks = chunk_context_by_sentences(context, target_chunks)
    
    # Ensure minimum chunk size (at least 20 characters)
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 20]
    
    logger.info(f"Final chunking result: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        logger.debug(f"Chunk {i+1}: {chunk[:100]}..." if len(chunk) > 100 else f"Chunk {i+1}: {chunk}")
    
    return chunks
