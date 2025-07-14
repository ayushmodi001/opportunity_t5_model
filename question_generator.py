import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ONNX-accelerated T5 for Question Generation ---
MODEL_NAME = "allenai/t5-small-squad2-question-generation"
ONNX_PATH = Path("onnx_models")

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Load ONNX model or convert if it doesn't exist
try:
    if not (ONNX_PATH / "encoder_model.onnx").exists():
        logger.info(f"ONNX model not found. Converting {MODEL_NAME} to ONNX format...")
        # Load PyTorch model
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        # Create ORTModelForSeq2SeqLM for conversion
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_NAME, export=True)
        # Save the ONNX model
        ort_model.save_pretrained(ONNX_PATH)
        logger.info(f"ONNX model saved to {ONNX_PATH}")
    
    # Load the ONNX model for inference
    onnx_model = ORTModelForSeq2SeqLM.from_pretrained(ONNX_PATH)
    logger.info("ONNX-accelerated T5 model loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load or convert the ONNX model: {e}")
    onnx_model = None

async def generate_questions(context: str, answers: list) -> list:
    """
    Generates one question for each (context, answer) pair using an
    ONNX-accelerated T5 model.
    """
    if not onnx_model:
        logger.error("Question generation model is not available.")
        return []

    logger.info(f"Generating questions for {len(answers)} answers...")
    qa_pairs = []
    generated_questions = set()

    for answer in answers:
        try:
            # Format input for T5 to be more explicit about the task
            input_text = f"generate question: answer: {answer} context: {context}"
            
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate question
            outputs = onnx_model.generate(
                **inputs,
                max_length=64,
                num_beams=5, # Increased beams for potentially better quality
                early_stopping=True
            )
            
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # The model often prepends "question: ", so we remove it.
            if question.startswith("question: "):
                question = question.replace("question: ", "", 1)

            question = question.strip()

            # Validate that the output is a question
            if not question.endswith('?'):
                logger.warning(f"Generated output is not a valid question, skipping: '{question}'")
                continue

            if question not in generated_questions:
                qa_pairs.append({"question": question, "answer": answer})
                generated_questions.add(question)
            else:
                logger.warning(f"Skipping duplicate question generated: '{question}'")
            
        except Exception as e:
            logger.error(f"Error generating question for answer '{answer}': {e}")
            continue
            
    logger.info(f"Generated {len(qa_pairs)} unique question-answer pairs.")
    return qa_pairs
