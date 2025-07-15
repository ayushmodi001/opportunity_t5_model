from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # Temporarily removed
from pydantic import BaseModel
from typing import List
import logging
import uvicorn
from config import BEARER_TOKEN, MISTRAL_API_KEY # Import from config
from context_fetcher import fetch_context
from summarizer import summarize_text
from answer_extractor import extract_answers
from question_generator import generate_questions
from distractor_generator import generate_distractors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Technical MCQ Generator",
    description="An API to generate technical multiple-choice questions.",
    version="1.0.0"
)

# security = HTTPBearer() # Temporarily removed
# BEARER_TOKEN is now imported from config

# def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)): # Temporarily removed
#     """Dependency to validate the bearer token."""
#     if not BEARER_TOKEN or credentials.credentials != BEARER_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect bearer token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     return credentials.credentials

class TopicsRequest(BaseModel):
    topics: List[str]

@app.post("/generate-mcqs/", response_model=dict)
async def generate_mcqs_endpoint(
    request: TopicsRequest,
    # token: str = Depends(get_current_user) # Temporarily removed
):
    """
    Endpoint to generate MCQs from a list of technical topics.
    """
    mcqs = []
    for topic in request.topics:
        try:
            logger.info(f"Processing topic: {topic}")

            # 1. Fetch Context
            context = await fetch_context(topic)
            logger.info(f"Fetched context for {topic}")

            # 2. Summarize if needed
            if len(context.split()) > 800:
                context = await summarize_text(context)
                logger.info(f"Summarized context for {topic}")

            # 3. Extract Answers
            answers = await extract_answers(context)
            logger.info(f"Extracted answers for {topic}: {answers}")

            if not answers:
                logger.warning(f"No answers extracted for {topic}. Skipping.")
                continue

            # 4. Generate Questions (targeting 10 questions)
            questions = await generate_questions(context, answers, target_questions=10)
            logger.info(f"Generated {len(questions)} questions for {topic}")

            # 5. Generate Distractors and format MCQs
            for q in questions:
                correct_answer = q["answer"]
                distractors = await generate_distractors(context, correct_answer)
                
                mcqs.append({
                    "topic": topic,
                    "question": q["question"],
                    "correct_answer": correct_answer,
                    "distractors": distractors,
                })

        except Exception as e:
            logger.error(f"Error processing topic {topic}: {e}")
            # Optionally, you can add a specific error message to the response
            # for the failed topic.
            continue
            
    if not mcqs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not generate any MCQs for the provided topics."
        )

    return {"questions": mcqs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
