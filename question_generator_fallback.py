import logging
import re
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import transformers, fall back to mock if not available
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available - using mock question generation")

# Simplified context chunker
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

# Mock question generation patterns
QUESTION_PATTERNS = [
    "What is {answer}?",
    "Which concept describes {answer}?",
    "What technology is {answer}?",
    "Which term refers to {answer}?",
    "What does {answer} represent?",
    "Which component is {answer}?",
    "What framework is {answer}?",
    "Which language is {answer}?",
    "What method involves {answer}?",
    "Which process uses {answer}?",
]

def generate_mock_question(chunk: str, answer: str) -> str:
    """Generate a mock question using simple patterns"""
    import random
    
    # Choose a random pattern
    pattern = random.choice(QUESTION_PATTERNS)
    question = pattern.format(answer=answer)
    
    # Try to make it more contextual
    if "programming" in chunk.lower():
        patterns = [
            f"What programming concept is {answer}?",
            f"Which programming language is {answer}?",
            f"What feature does {answer} provide?",
        ]
        question = random.choice(patterns)
    elif "development" in chunk.lower():
        patterns = [
            f"What development tool is {answer}?",
            f"Which development framework is {answer}?",
            f"What methodology involves {answer}?",
        ]
        question = random.choice(patterns)
    
    return question

# Load T5 model if available
if TRANSFORMERS_AVAILABLE:
    MODEL_NAME = "t5-small"
    try:
        logger.info(f"Loading T5 model: {MODEL_NAME}")
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        logger.info("T5 model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load T5 model: {e}")
        tokenizer = None
        model = None
        TRANSFORMERS_AVAILABLE = False
else:
    tokenizer = None
    model = None

async def generate_questions_from_chunk(chunk: str, answers: List[str]) -> List[dict]:
    """Generates questions from a single context chunk using extracted answers."""
    qa_pairs = []
    generated_questions = set()

    for answer in answers:
        try:
            # Check if the answer appears in this chunk (case insensitive)
            if answer.lower() not in chunk.lower():
                continue
            
            if TRANSFORMERS_AVAILABLE and model and tokenizer:
                # Use T5 model
                input_text = f"question: {chunk} answer: {answer}"
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                
                outputs = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=3,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.8
                )
                
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # Use mock generation
                question = generate_mock_question(chunk, answer)
            
            question = question.strip()

            # Ensure it's a proper question
            if not question.endswith('?'):
                question += '?'

            if question not in generated_questions and len(question) > 10:
                qa_pairs.append({"question": question, "answer": answer})
                generated_questions.add(question)
            
        except Exception as e:
            logger.error(f"Error generating question for answer '{answer}' in chunk: {e}")
            continue
            
    return qa_pairs

async def generate_questions(context: str, answers: list, target_questions: int = 10) -> list:
    """
    Generates diverse questions by breaking context into chunks and generating 
    questions from each chunk using different answer combinations.
    """
    logger.info(f"Generating {target_questions} questions from context using {len(answers)} answers...")
    
    # Break context into chunks for diverse question generation
    chunks = simple_chunk_context(context, target_questions)
    
    all_qa_pairs = []
    generated_questions = set()
    
    # Distribute answers across chunks to ensure variety
    for i, chunk in enumerate(chunks):
        if len(all_qa_pairs) >= target_questions:
            break
            
        # Use different answer subsets for each chunk to increase diversity
        chunk_answers = answers[i % len(answers):] + answers[:i % len(answers)]
        chunk_answers = chunk_answers[:min(3, len(answers))]  # Limit to 3 answers per chunk
        
        logger.info(f"Processing chunk {i+1}/{len(chunks)} with answers: {chunk_answers}")
        
        chunk_qa_pairs = await generate_questions_from_chunk(chunk, chunk_answers)
        
        # Add unique questions only
        for qa_pair in chunk_qa_pairs:
            if (qa_pair["question"] not in generated_questions and 
                len(all_qa_pairs) < target_questions):
                all_qa_pairs.append(qa_pair)
                generated_questions.add(qa_pair["question"])
    
    # If we still don't have enough questions, generate more with mock patterns
    if len(all_qa_pairs) < target_questions:
        logger.info(f"Only generated {len(all_qa_pairs)} questions, adding mock questions...")
        
        remaining_needed = target_questions - len(all_qa_pairs)
        for i in range(remaining_needed):
            if i < len(answers):
                answer = answers[i % len(answers)]
                question = generate_mock_question(context, answer)
                
                if question not in generated_questions:
                    all_qa_pairs.append({"question": question, "answer": answer})
                    generated_questions.add(question)
            
    logger.info(f"Generated {len(all_qa_pairs)} unique question-answer pairs.")
    return all_qa_pairs[:target_questions]
