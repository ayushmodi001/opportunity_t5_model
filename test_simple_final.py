#!/usr/bin/env python3
"""
Simplified test to verify core functionality
"""
import asyncio

# Test 1: Simple chunking
def test_chunking():
    print("1️⃣ Testing Context Chunking...")
    
    sample_text = """
    Python is a programming language. It was created by Guido van Rossum.
    Python supports multiple paradigms. It emphasizes code readability.
    The language has an extensive standard library. Popular frameworks include Django.
    Python is used in web development and data science.
    """
    
    # Simple sentence splitting
    import re
    sentences = re.split(r'[.!?]+', sample_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print(f"✅ Split into {len(sentences)} sentences:")
    for i, sentence in enumerate(sentences):
        print(f"   {i+1}: {sentence}")
    
    return sentences

# Test 2: Answer extraction without external libraries
def test_answer_extraction():
    print("\n2️⃣ Testing Answer Extraction...")
    
    text = "Python is a programming language created by Guido van Rossum in 1991"
    
    # Simple word extraction
    import re
    words = re.findall(r'\b[A-Za-z]{3,}\b', text)
    
    # Filter common words
    stop_words = {'the', 'is', 'in', 'by', 'and', 'or', 'but', 'was', 'were', 'are'}
    answers = [word for word in words if word.lower() not in stop_words]
    
    print(f"✅ Extracted answers: {answers[:5]}")
    return answers

# Test 3: Mock T5 question generation
async def test_question_generation():
    print("\n3️⃣ Testing Question Generation...")
    
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        print("Loading T5 model...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        
        context = "Python is a programming language created by Guido van Rossum"
        answer = "Python"
        
        input_text = f"question: {context} answer: {answer}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=3,
            early_stopping=True
        )
        
        question = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Generated question: {question}")
        return [{"question": question, "answer": answer}]
        
    except Exception as e:
        print(f"❌ T5 model error: {e}")
        # Fallback to mock questions
        mock_questions = [
            {"question": "What is Python?", "answer": "Python"},
            {"question": "Who created Python?", "answer": "Guido van Rossum"}
        ]
        print(f"✅ Using mock questions: {mock_questions}")
        return mock_questions

# Test 4: Mock distractor generation
async def test_distractor_generation():
    print("\n4️⃣ Testing Distractor Generation...")
    
    correct_answer = "Python"
    
    # Simple mock distractors
    distractors = [
        "Java",
        "C++", 
        "JavaScript"
    ]
    
    print(f"✅ Generated distractors for '{correct_answer}': {distractors}")
    return distractors

async def main():
    print("🧪 Simplified MCQ Testing")
    print("=" * 40)
    
    # Run tests
    sentences = test_chunking()
    answers = test_answer_extraction()
    questions = await test_question_generation()
    distractors = await test_distractor_generation()
    
    # Create final MCQ
    print("\n5️⃣ Creating Final MCQ...")
    
    if questions and distractors:
        mcq = {
            "topic": "Python programming",
            "question": questions[0]["question"],
            "correct_answer": questions[0]["answer"],
            "distractors": distractors
        }
        
        print("✅ Sample MCQ created:")
        print(f"   Topic: {mcq['topic']}")
        print(f"   Question: {mcq['question']}")
        print(f"   Correct Answer: {mcq['correct_answer']}")
        print(f"   Distractors: {', '.join(mcq['distractors'])}")
        
        # Save to file
        import json
        with open("simple_test_result.json", "w") as f:
            json.dump({"questions": [mcq]}, f, indent=2)
        print("💾 Saved to simple_test_result.json")
        
    print("\n🎉 Simplified test completed!")

if __name__ == "__main__":
    asyncio.run(main())
