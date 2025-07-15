import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from context_chunker import chunk_context_smart

async def test_context_chunker():
    """Test the context chunker functionality"""
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
    
    print("Original context length:", len(sample_context.split()))
    
    chunks = chunk_context_smart(sample_context, target_chunks=5)
    
    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Length: {len(chunk.split())} words")
        print(f"Content: {chunk}")

if __name__ == "__main__":
    asyncio.run(test_context_chunker())
