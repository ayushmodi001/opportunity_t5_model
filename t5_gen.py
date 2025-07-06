import signal
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re
from mistralai.client import MistralClient
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Generation took too long!")

# Set timeout handler
signal.signal(signal.SIGALRM, timeout_handler)

def get_context_from_mistral(topic):
    """Get technical context from Mistral API"""
    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    
    current_date = "2025-07-06"  # Using current date to ensure unique content
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical expert providing detailed explanations about programming concepts. "
                "Generate unique, non-repetitive content each time. Include detailed technical examples."
            )
        },
        {
            "role": "user",
            "content": (
                f"Today's date is {current_date}. Generate a unique, comprehensive technical explanation about {topic}.\n\n"
                f"Provide a detailed explanation covering:\n"
                f"1. Core concepts and fundamentals with code examples\n"
                f"2. Key features and capabilities with practical use cases\n"
                f"3. Best practices and design patterns with real-world scenarios\n"
                f"4. Technical advantages and common pitfalls to avoid\n"
                f"5. Advanced techniques and optimization strategies\n\n"
                f"Requirements:\n"
                f"- Make this explanation unique and different from standard documentation\n"
                f"- Include practical code examples\n"
                f"- Provide in-depth technical details\n"
                f"- Format with clear sections and bullet points\n"
                f"- Length: 500-800 words\n"  # Reduced length for faster processing
                f"- Focus on advanced concepts and real-world applications"
            )
        }
    ]
    
    try:
        chat_response = client.chat(
            model="mistral-tiny",  # Using tiny model for faster responses
            messages=messages,
            max_tokens=1000,  # Limit token count
            temperature=0.7    # Lower temperature for more focused output
        )
        content = chat_response.choices[0].message.content
        print("\nContext received from Mistral:")
        print("-" * 40)
        print(content[:500] + "..." if len(content) > 500 else content)
        print("-" * 40)
        return content
    except Exception as e:
        print(f"Error with Mistral API: {str(e)}")
        # Fallback context in case of API issues
        return """
Python is a high-level, interpreted programming language known for its simplicity and readability.
Key features include:
1. Dynamic typing and automatic memory management
2. Comprehensive standard library ("batteries included")
3. Rich ecosystem of third-party packages
4. Support for multiple programming paradigms (procedural, object-oriented, functional)
5. Extensive built-in data structures
Best practices include using clear naming conventions, following PEP 8 style guide, and writing docstrings.
"""

def format_mcq(raw_output):
    """Format the raw model output into properly structured MCQs."""
    # Basic cleanup
    text = raw_output.strip()
    
    # Split multiple MCQs if present
    mcq_sections = re.split(r'\n\s*(?=Question:\s*)', text)
    
    formatted_mcqs = []
    
    for mcq in mcq_sections:
        if not mcq.strip():
            continue
            
        # Extract question
        question_match = re.search(r'Question:\s*(.+?)(?=[A|B|C|D]\)|$)', mcq, re.IGNORECASE | re.DOTALL)
        
        # Extract options
        options_matches = re.findall(r'([A-D])\)\s*([^A-D\n][^\n]+)', mcq, re.IGNORECASE)
        
        # Extract answer
        answer_match = re.search(r'(?:Answer|Correct Answer|correct)[^A-D]*([A-D])', mcq, re.IGNORECASE)
        
        if question_match and options_matches:
            formatted_mcq = []
            formatted_mcq.append(f"Question: {question_match.group(1).strip()}")
            formatted_mcq.append("")
            
            for opt, text in options_matches:
                formatted_mcq.append(f"{opt}) {text.strip()}")
            
            if answer_match:
                formatted_mcq.append("")
                formatted_mcq.append(f"Correct Answer: {answer_match.group(1)}")
            
            formatted_mcqs.append("\n".join(formatted_mcq))
    
    return "\n\n" + "-" * 60 + "\n\n".join(formatted_mcqs) + "\n" + "-" * 60

def prepare_mcq_prompt(context, sample_context, sample_mcq):
    template = f"""
Generate exactly 10 multiple-choice questions based on the following context. 
NOTE: The sample context and MCQ provided are for format reference only. Generate questions based on the new context provided.

Rules for each MCQ:

1. Question Requirements:
   - Start with "Question: "
   - Test deep technical understanding
   - Be clear and unambiguous
   - Focus on important concepts

2. Options Requirements:
   - Label options as "A) ", "B) ", "C) ", and "D) "
   - Make each option a complete, grammatically correct sentence
   - Ensure parallel structure
   - Make distractors plausible but clearly incorrect
   - Make options distinct and non-overlapping

3. Format Requirements:
   - Include one blank line after each question
   - Include one option per line
   - End each MCQ with "Correct Answer: [A/B/C/D]"
   - Add a blank line between MCQs

New Context to use for generating MCQs:
{context}

Sample Context (for format reference only):
{sample_context}

Sample MCQ Format:
{sample_mcq}

Now generate 10 new MCQs using the new context, following this exact format:
"""
    return template

# Sample context and MCQ for reference
sample_context = '''
Python is a high-level, interpreted programming language known for its versatility and readability.
It features dynamic typing, automatic memory management, and comprehensive standard libraries.
Python's object-oriented features and functional programming capabilities make it powerful for diverse applications.
'''

sample_mcq = '''
Question: What is the primary feature that makes Python suitable for rapid development?
A) Python uses manual memory management for better performance.
B) Python's dynamic typing and automatic memory management simplify development.
C) Python requires explicit type declarations for all variables.
D) Python only supports procedural programming paradigms.
Correct Answer: B
'''

# Initialize model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Get context from Mistral
print("Getting context from Mistral API...")
context = get_context_from_mistral("Python programming language features and best practices")

# Prepare the prompt
prompt = prepare_mcq_prompt(context, sample_context, sample_mcq)

# Generate MCQs
print("\nGenerating MCQs...")
inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)  # Reduced input length
inputs = {k: v.to(device) for k, v in inputs.items()}

try:
    # Set timeout for 2 minutes
    signal.alarm(120)
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_length=800,        # Significantly reduced
        num_beams=2,           # Reduced beam search
        temperature=0.6,       # Lower temperature for more focused outputs
        top_p=0.9,
        do_sample=True,
        no_repeat_ngram_size=3,
        early_stopping=True,
        length_penalty=1.0,    # Neutral length penalty
        repetition_penalty=1.2,
        num_return_sequences=1,
        min_length=200,        # Reduced minimum length
        max_new_tokens=600,    # Explicit limit on new tokens
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generation_time = time.time() - start_time
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    
    # Disable timeout
    signal.alarm(0)
    
except TimeoutException:
    print("\nGeneration timed out after 2 minutes. Using partial output...")
    # Return whatever was generated so far
    
except Exception as e:
    print(f"\nError during generation: {str(e)}")
    raise

# Decode and format the generated MCQs
generated_mcqs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

print("\nRaw output from T5:")
print("-" * 40)
print(generated_mcqs[:500] + "..." if len(generated_mcqs) > 500 else generated_mcqs)
print("-" * 40)

formatted_mcqs = format_mcq(generated_mcqs)

print("\nFormatted MCQs:")
print(formatted_mcqs)