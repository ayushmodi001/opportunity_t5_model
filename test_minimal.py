#!/usr/bin/env python3
"""
Test the system with minimal dependencies
"""
import asyncio
import json

async def test_minimal_system():
    """Test the MCQ system with minimal dependencies"""
    print("🧪 Testing Minimal MCQ System")
    print("=" * 35)
    
    # Test basic imports
    print("1️⃣ Testing basic imports...")
    try:
        import config
        print("✅ Config imported")
        
        # Test context fetching (will use fallback)
        from context_fetcher import fetch_context
        print("✅ Context fetcher imported")
        
        # Test answer extraction (basic NLP)
        from answer_extractor import extract_answers
        print("✅ Answer extractor imported")
        
        # Test question generation (will use fallback if T5 not available)
        from question_generator_fallback import generate_questions
        print("✅ Question generator imported")
        
        # Test distractor generation
        from distractor_generator import generate_distractors
        print("✅ Distractor generator imported")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test the complete flow
    print("\n2️⃣ Testing MCQ generation flow...")
    try:
        # Sample data
        topic = "Python programming"
        sample_context = """
        Python is a high-level programming language known for its simplicity and readability.
        It was created by Guido van Rossum and first released in 1991.
        Python supports multiple programming paradigms including object-oriented and functional programming.
        The language is widely used in web development, data science, and automation.
        Popular frameworks include Django for web development and NumPy for scientific computing.
        """
        
        # Extract answers
        answers = await extract_answers(sample_context)
        print(f"✅ Extracted {len(answers)} answers: {answers[:5]}")
        
        # Generate questions
        questions = await generate_questions(sample_context, answers, target_questions=5)
        print(f"✅ Generated {len(questions)} questions")
        
        # Generate distractors for first question
        if questions:
            distractors = await generate_distractors(sample_context, questions[0]["answer"])
            print(f"✅ Generated distractors: {distractors}")
        
        # Create final MCQs
        mcqs = []
        for qa in questions[:3]:  # Test with 3 questions
            distractors = await generate_distractors(sample_context, qa["answer"])
            mcq = {
                "topic": topic,
                "question": qa["question"],
                "correct_answer": qa["answer"],
                "distractors": distractors
            }
            mcqs.append(mcq)
        
        print(f"✅ Created {len(mcqs)} complete MCQs")
        
        # Save results
        result = {"questions": mcqs}
        with open("minimal_test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Display sample
        if mcqs:
            sample = mcqs[0]
            print(f"\n📝 Sample MCQ:")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['correct_answer']}")
            print(f"Distractors: {', '.join(sample['distractors'])}")
        
        print("\n🎉 Minimal system test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in MCQ generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_minimal_system())
    if success:
        print("\n✅ System is working with available dependencies!")
        print("You can now start the server with: python -m uvicorn main:app --reload")
    else:
        print("\n❌ System test failed - check error messages above")
