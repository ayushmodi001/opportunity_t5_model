"""
Simple test to verify the context chunking works without external dependencies
"""

def simple_chunk_by_sentences(context: str, target_chunks: int = 10):
    """Simple sentence-based chunking without spaCy"""
    import re
    
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', context)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= target_chunks:
        return sentences
    
    sentences_per_chunk = max(1, len(sentences) // target_chunks)
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk = ". ".join(chunk_sentences) + "."
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks[:target_chunks]

# Test the function
sample_context = """
Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models.
These algorithms enable computers to learn and make decisions from data without being explicitly programmed for every task.
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
Supervised learning uses labeled datasets to train algorithms to classify data or predict outcomes accurately.
Neural networks are computing systems inspired by biological neural networks that constitute animal brains.
Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns.
Popular frameworks for machine learning include TensorFlow, PyTorch, and Scikit-learn.
These frameworks provide tools and libraries that make it easier to develop machine learning applications.
"""

print("Testing simple context chunking:")
print("Original context length:", len(sample_context.split()), "words")

chunks = simple_chunk_by_sentences(sample_context, target_chunks=5)

print(f"\nGenerated {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"Length: {len(chunk.split())} words")
    print(f"Content: {chunk[:100]}...")
