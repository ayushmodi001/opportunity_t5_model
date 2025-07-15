import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified context chunker without spaCy
def simple_chunk_context(context: str, target_chunks: int = 10) -> List[str]:
    """Break context into chunks using simple sentence splitting"""
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', context)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
    
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

# Load T5 model for question generation
MODEL_NAME = "t5-small"
print("Loading T5 model...")
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    print("✅ T5 model loaded successfully")
except Exception as e:
    print(f"❌ Error loading T5 model: {e}")
    tokenizer = None
    model = None

async def generate_questions_simple(context: str, answers: list, target_questions: int = 10) -> list:
    """Generate questions using simple chunking approach"""
    if not model or not tokenizer:
        print("❌ Model not available")
        return []
    
    print(f"Generating {target_questions} questions...")
    
    # Break context into chunks
    chunks = simple_chunk_context(context, target_questions)
    print(f"Created {len(chunks)} chunks")
    
    all_qa_pairs = []
    generated_questions = set()
    
    for i, chunk in enumerate(chunks):
        if len(all_qa_pairs) >= target_questions:
            break
            
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Use relevant answers for this chunk
        chunk_answers = [ans for ans in answers if ans.lower() in chunk.lower()]
        
        if not chunk_answers:
            # If no answers found in chunk, use first available answer
            chunk_answers = answers[:1]
        
        for answer in chunk_answers[:2]:  # Limit to 2 answers per chunk
            if len(all_qa_pairs) >= target_questions:
                break
                
            try:
                # Format input for T5
                input_text = f"question: {chunk} answer: {answer}"
                
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                
                # Generate question
                outputs = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.8
                )
                
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)
                question = question.strip()
                
                # Ensure it's a proper question
                if not question.endswith('?'):
                    question += '?'
                
                if (len(question) > 10 and 
                    question not in generated_questions):
                    all_qa_pairs.append({"question": question, "answer": answer})
                    generated_questions.add(question)
                    print(f"   Generated: {question}")
                    
            except Exception as e:
                print(f"   Error generating question for '{answer}': {e}")
                continue
    
    print(f"✅ Generated {len(all_qa_pairs)} unique questions")
    return all_qa_pairs

# Test the functionality
async def test_question_generation():
    sample_context = """
    Python is a high-level programming language known for its simplicity and readability.
    It was created by Guido van Rossum and first released in 1991.
    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    The language emphasizes code readability with its notable use of significant whitespace.
    Python's standard library is extensive and includes modules for various tasks.
    Popular frameworks include Django for web development and NumPy for scientific computing.
    Python is widely used in web development, data science, artificial intelligence, and automation.
    """
    
    sample_answers = ["Python", "Guido van Rossum", "1991", "Django", "NumPy", "programming language"]
    
    print("🧪 Testing Question Generation")
    print("=" * 40)
    
    questions = await generate_questions_simple(sample_context, sample_answers, target_questions=5)
    
    print(f"\n📝 Results:")
    for i, qa in enumerate(questions):
        print(f"Q{i+1}: {qa['question']}")
        print(f"A{i+1}: {qa['answer']}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_question_generation())
