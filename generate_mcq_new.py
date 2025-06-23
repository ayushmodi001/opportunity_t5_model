import os
import json
import random
import logging
import re
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Set
import nltk
from difflib import SequenceMatcher
from dotenv import load_dotenv
from transformers import T5ForConditionalGeneration, T5Tokenizer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcq_generator.log'),
        logging.StreamHandler()
    ]
)

class RateLimiter:
    def __init__(self, calls_per_minute: int = 20):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0
        
    async def wait(self):
        """Wait if necessary to maintain rate limits"""
        now = time.time()
        time_since_last = now - self.last_call
        if time_since_last < self.interval:
            wait_time = self.interval - time_since_last
            await asyncio.sleep(wait_time)
        self.last_call = time.time()

class AsyncMistralWrapper:
    def __init__(self, client):
        self.client = client
        self.rate_limiter = RateLimiter(calls_per_minute=20)  # More conservative rate limit
        self.max_retries = 3
        self.base_delay = 2  # Longer base delay for exponential backoff

    async def generate_text(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Async wrapper for Mistral text generation with retries and exponential backoff"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.wait()
                response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model="mistral-tiny",
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)  # Add jitter
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {str(e)}")
                    try:
                        await asyncio.sleep(delay)
                    except asyncio.CancelledError:
                        logger.warning("Operation cancelled during backoff")
                        raise
                        
        logger.error(f"All API call attempts failed after {self.max_retries} retries")
        raise last_error if last_error else Exception("API call failed")

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"Failed to download NLTK data: {e}")

# Initialize logger
logger = logging.getLogger(__name__)

class MCQGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv(override=True)
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
        if not self.mistral_api_key:
            raise ValueError("Please ensure MISTRAL_API_KEY is set in your .env file")
        
        try:
            self.client = MistralClient(api_key=self.mistral_api_key.strip('"\''))
            self.mistral = AsyncMistralWrapper(self.client)
            
            # Use ONNX model for faster inference
            model_path = "/tmp/mcq_model" if os.path.exists("/tmp") else "./mcq_model"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            
            # Download model only if not exists (for Lambda cold starts)
            if not os.path.exists(os.path.join(model_path, "model.onnx")):
                print("Downloading optimized model...")
                self._download_model(model_path)
            
            # Load ONNX model
            import onnxruntime as ort
            self.ort_session = ort.InferenceSession(
                os.path.join(model_path, "model.onnx"),
                providers=['CPUExecutionProvider']  # Use CPU for Lambda
            )
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            print("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing: {e}")
            raise
        
        self.logger = logging.getLogger(__name__)
        self.current_topic = None
        self.used_questions = set()
    
    def _download_model(self, model_path: str):
        """Download the optimized model from S3 or similar storage"""
        # TODO: Replace with your S3 bucket details
        import boto3
        s3 = boto3.client('s3')
        try:
            s3.download_file(
                'your-bucket-name',
                'mcq_model/model.onnx',
                os.path.join(model_path, "model.onnx")
            )
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise

    def get_focused_answer_prompt(self, question: str, context: str) -> str:
        """Generate a focused prompt for answer generation"""
        return f"""Based on this context, provide a clear, direct answer to the question in 5-10 words.

Context: {context}
Question: {question}

Requirements:
1. Must be EXACTLY 5-10 words long
2. Must be factually correct based on context
3. Must use simple, clear technical terms
4. Must be a complete sentence
5. Must end with proper punctuation
6. Must start with a capital letter
7. Must not use "it depends" or hedging language
8. Must not include explanations or qualifiers

Examples of good answers:
- Neural networks learn patterns through weighted connection adjustments.
- Binary search divides the search space repeatedly.
- Databases ensure data consistency through ACID properties.
- TCP guarantees reliable packet delivery between hosts.

Generate a concise 5-10 word answer:"""

    def get_improved_distractors_prompt(self, question: str, context: str, correct_answer: str) -> str:
        """Generate an improved prompt for distractor generation"""
        return f"""Create 3 plausible but incorrect answers for this question.

Context: {context}
Question: {question}
Correct Answer: {correct_answer}

Requirements for each distractor:
1. Must be exactly {len(correct_answer.split())} words long (same as correct answer)
2. Must be verifiably incorrect based on the context
3. Must use the same technical level as the correct answer
4. Must be a complete sentence with proper punctuation
5. Must be grammatically consistent with the question
6. No joke answers or nonsensical responses
7. No "all/none of the above" type options
8. Each distractor must reflect a specific misconception or error

Format guide:
- Write each distractor on its own line
- No numbering, bullets, or prefixes
- Do not use "because" or explanatory clauses
- Match the sentence structure of the correct answer

Just write the 3 distractors, separated by newlines, without any additional text."""
    
    async def get_context_for_topic(self, topic: str) -> str:
        """Generate or retrieve context for a given topic using Mistral"""
        try:
            # More specific prompt for better context
            prompt = f"""Generate a comprehensive explanation about {topic}. Include:
            1. Key concepts and principles
            2. Important techniques or methods
            3. Real-world applications
            4. Common challenges or considerations
            Keep it detailed but under 500 words. Focus on factual information."""
            
            response = await self.mistral.generate_text(prompt)
            # Updated response handling
            return response
        except Exception as e:
            self.logger.error(f"Error generating context with Mistral: {e}")
            # More informative fallback context based on topic
            return self._get_fallback_context(topic)
            
    def _get_fallback_context(self, topic: str) -> str:
        """Provide a meaningful fallback context when Mistral API fails"""
        base_contexts = {
            "Algorithms": """Algorithms are step-by-step procedures for solving computational problems. They include searching, sorting, and graph algorithms. Key concepts include time complexity, space complexity, and algorithmic efficiency. Common algorithms include binary search, quicksort, and depth-first search. Algorithm design techniques include divide-and-conquer, dynamic programming, and greedy approaches.""",
            "Data Structures": """Data structures are specialized formats for organizing and storing data. Basic structures include arrays, linked lists, stacks, and queues. Advanced structures include trees, graphs, and hash tables. Each structure has specific use cases, advantages, and trade-offs in terms of time and space complexity.""",
            "Database Management": """Database management involves organizing, storing, and retrieving data efficiently. Key concepts include ACID properties, normalization, and SQL. Database types include relational, NoSQL, and distributed databases. Important aspects include query optimization, indexing, and transaction management."""
        }
        
        # Try to find a close match in our base contexts
        for key in base_contexts:
            if key.lower() in topic.lower():
                return base_contexts[key]
        
        # Generate a semi-structured fallback for unknown topics
        return f"""The field of {topic} involves several key areas:
1. Core principles and fundamental concepts
2. Methods and techniques commonly used
3. Practical applications and implementations
4. Current trends and modern approaches
This field continues to evolve with new developments and applications."""

    async def generate_question(self, context: str) -> str:
        """Generate a question using the optimized ONNX model"""
        try:
            context_summary = self._extract_key_concepts(context)
            input_text = f"generate question: {context_summary}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=128,  # Reduced from 512
                truncation=True,
                return_tensors="pt"
            )
            
            # Run inference with ONNX
            ort_inputs = {
                'input_ids': inputs['input_ids'].numpy(),
                'attention_mask': inputs['attention_mask'].numpy()
            }
            
            ort_outputs = self.ort_session.run(None, ort_inputs)
            
            # Decode output
            question = self.tokenizer.decode(
                ort_outputs[0][0],
                skip_special_tokens=True
            )
            
            question = self.clean_question(question)
            if self._validate_question(question):
                return question
            
            # Simplified fallback - no multiple attempts to save compute
            return await self._generate_fallback_question(context)
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return ""

    def get_context_sync(self, topic: str) -> str:
        """Synchronous wrapper for get_context_for_topic"""
        return asyncio.get_event_loop().run_until_complete(self.get_context_for_topic(topic))
        
    def save_mcqs_to_file(self, mcqs: List[Dict], filename: str) -> str:
        """Save MCQs to a JSON file and return the file path"""
        try:
            # Create mcqs directory if it doesn't exist
            output_dir = os.path.join(os.getcwd(), "mcqs")
            os.makedirs(output_dir, exist_ok=True)
            
            # Full path for output file
            output_file = os.path.join(output_dir, filename)
            
            # Save MCQs to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(mcqs, f, ensure_ascii=False, indent=2)
                
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving MCQs to file: {e}")
            raise

    async def test_mcq_quality(self, topic: str) -> Dict:
        """Test the quality of generated MCQs"""
        try:
            self.current_topic = topic
            context = await self.get_context_for_topic(topic)
            
            # Generate and validate question
            question = await self.generate_question(context)
            if not self._validate_question(question):
                return {
                    "status": "error",
                    "message": "Failed to generate valid question",
                    "topic": topic
                }
            
            # Generate and validate options
            options = await self.generate_answer_and_distractors(question, context)
            if len(options) != 4:
                return {
                    "status": "error",
                    "message": "Failed to generate valid options",
                    "topic": topic
                }
            
            # Calculate quality metrics
            question_metrics = {
                "length": len(question.split()),
                "has_context": bool(re.search(r'\b' + re.escape(topic) + r'\b', question, re.IGNORECASE)),
                "ends_with_question": question.strip().endswith('?')
            }
            
            option_metrics = {
                "lengths": [len(opt.split()) for opt in options],
                "similarity_ratios": [
                    SequenceMatcher(None, opt1.lower(), opt2.lower()).ratio()
                    for i, opt1 in enumerate(options)
                    for j, opt2 in enumerate(options)
                    if i < j
                ]
            }
            
            return {
                "status": "success",
                "topic": topic,
                "question": question,
                "correct_answer": options[0],
                "distractors": options[1:],
                "quality_metrics": {
                    "question": question_metrics,
                    "options": option_metrics
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in MCQ quality test: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "topic": topic
            }

    def _standardize_option(self, text: str) -> str:
        """Standardize an option's format and length"""
        # Clean and standardize text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.capitalize()
        
        # Remove common prefixes
        text = re.sub(r'^(The|A|An)\s+', '', text, flags=re.IGNORECASE)
        
        # Target length: 5-10 words
        words = text.split()
        if len(words) > 10:
            text = ' '.join(words[:10])
        elif len(words) < 5:
            return ""  # Too short to be valid
            
        # Add period if not present
        if not text.endswith(('.', '?', '!')):
            text += '.'
            
        return text

    async def _generate_fallback_question(self, context: str) -> str:
        """Generate a fallback question when the main generation fails"""
        try:
            fallback_prompt = f"""
            Generate a simple technical question about {self.current_topic} that:
            1. Is 15-25 words long
            2. Tests basic understanding
            3. Ends with a question mark
            4. Uses the following context:
            "{context}"
            
            Question:"""
            
            response = await self.mistral.generate_text(
                fallback_prompt,
                max_tokens=75,
                temperature=0.5
            )
            
            question = self._clean_question(response.strip())
            return question if self._validate_question(question) else ""
        except Exception as e:
            self.logger.error(f"Error in fallback question generation: {str(e)}")
            return ""

def main():
    """Main function to run the MCQ generator"""
    async def run_async():
        # Initialize generator
        generator = MCQGenerator()
        
        # Get topic from user
        print("\nMCQ Generator")
        print("="*50)
        print("You can generate MCQs for any topic! Examples: Python Programming, Machine Learning,")
        print("Quantum Physics, World History, or any other topic you're interested in.")
        print("="*50)
        
        selected_topic = input("\nEnter your desired topic: ").strip()
        
        print(f"\n{'='*60}")
        print(f"Generating MCQs for topic: {selected_topic}")
        print(f"{'='*60}\n")
        
        try:
            # Get context for the topic
            print("Generating context...")
            context = await generator.get_context_for_topic(selected_topic)
            print(f"\nGenerated context:\n{context}\n")
            
            # Get number of questions from user
            while True:
                try:
                    num_questions = int(input("\nHow many questions would you like to generate? (1-10): "))
                    if 1 <= num_questions <= 10:
                        break
                    print("Please enter a number between 1 and 10")
                except ValueError:
                    print("Please enter a valid number")
            
            # Generate MCQs
            print("\nGenerating MCQs...")
            mcqs = await generator.generate_mcqs(context, num_questions, selected_topic)
            
            # Save MCQs
            output_file = generator.save_mcqs_to_file(
                mcqs, 
                f"mcqs_{selected_topic.lower().replace(' ', '_')}.json"
            )
            print(f"\nMCQs have been saved to: {output_file}")
            
            # Print generated MCQs
            print("\nGenerated MCQs:")
            for i, mcq in enumerate(mcqs, 1):
                print(f"\nQuestion {i}:")
                print(f"Context: {mcq['context']}")
                print(f"Q: {mcq['question']}")
                print("Options:")
                for j, option in enumerate(mcq['options']):
                    print(f"{chr(65+j)}. {option}")
                print(f"Correct Answer: {mcq['correct_answer']}")
                
        except Exception as e:
            logger.error(f"Error processing topic '{selected_topic}': {e}")
            
    # Run the async function using asyncio.run
    asyncio.run(run_async())

if __name__ == "__main__":
    main()
