#!/usr/bin/env python3
"""
Test individual components of the MCQ generation system
"""
import asyncio
import json
import sys
import os

async def test_individual_components():
    """Test each component separately"""
    
    print("🧪 Testing Individual Components")
    print("=" * 50)
    
    # Sample data for testing
    test_topic = "Python programming basics"
    sample_context = """
    Python is a high-level, interpreted programming language with dynamic semantics.
    Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.
    Python's simple, easy to learn syntax emphasizes readability and therefore reduces the cost of program maintenance.
    Python supports modules and packages, which encourages program modularity and code reuse.
    The Python interpreter and the extensive standard library are available in source or binary form without charge.
    """
    
    print(f"📚 Test Topic: {test_topic}")
    print(f"📝 Context Length: {len(sample_context.split())} words")
    
    # 1. Test Context Fetching
    print("\n1️⃣ Testing Context Fetching...")
    try:
        from context_fetcher import fetch_context
        
        # Note: This will make an actual API call to Mistral
        # context = await fetch_context(test_topic)
        # For testing, we'll use the sample context
        context = sample_context
        
        print(f"✅ Context available ({len(context.split())} words)")
        print(f"Sample: {context[:150]}...")
        
    except Exception as e:
        print(f"❌ Context fetching error: {e}")
        context = sample_context
        print("📝 Using sample context for testing")
    
    # 2. Test Answer Extraction
    print("\n2️⃣ Testing Answer Extraction...")
    try:
        from answer_extractor import extract_answers
        
        answers = await extract_answers(context)
        print(f"✅ Extracted {len(answers)} answers")
        print(f"Answers: {answers[:5]}...")  # Show first 5
        
    except Exception as e:
        print(f"❌ Answer extraction error: {e}")
        # Fallback answers
        answers = ["Python", "programming language", "interpreter", "modules", "packages"]
        print(f"📝 Using fallback answers: {answers}")
    
    # 3. Test Question Generation
    print("\n3️⃣ Testing Question Generation...")
    try:
        from question_generator import generate_questions
        
        questions = await generate_questions(context, answers, target_questions=5)
        print(f"✅ Generated {len(questions)} questions")
        
        for i, qa in enumerate(questions):
            print(f"   Q{i+1}: {qa['question']}")
            print(f"   A{i+1}: {qa['answer']}")
            print()
            
    except Exception as e:
        print(f"❌ Question generation error: {e}")
        questions = []
    
    # 4. Test Distractor Generation
    if questions:
        print("4️⃣ Testing Distractor Generation...")
        try:
            from distractor_generator import generate_distractors
            
            first_question = questions[0]
            distractors = await generate_distractors(context, first_question["answer"])
            
            print(f"✅ Generated distractors for '{first_question['answer']}':")
            for i, distractor in enumerate(distractors):
                print(f"   D{i+1}: {distractor}")
                
        except Exception as e:
            print(f"❌ Distractor generation error: {e}")
    
    # 5. Test Complete MCQ Generation
    print("\n5️⃣ Testing Complete MCQ Generation...")
    try:
        mcqs = []
        
        for q in questions[:3]:  # Test with first 3 questions
            try:
                from distractor_generator import generate_distractors
                distractors = await generate_distractors(context, q["answer"])
                
                mcq = {
                    "topic": test_topic,
                    "question": q["question"],
                    "correct_answer": q["answer"],
                    "distractors": distractors
                }
                mcqs.append(mcq)
                
            except Exception as e:
                print(f"   ⚠️ Error creating MCQ for '{q['answer']}': {e}")
        
        print(f"✅ Created {len(mcqs)} complete MCQs")
        
        # Save results
        result = {"questions": mcqs}
        with open("component_test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("💾 Results saved to component_test_results.json")
        
        # Display results
        print("\n📋 Final MCQs:")
        for i, mcq in enumerate(mcqs):
            print(f"\n--- MCQ {i+1} ---")
            print(f"Question: {mcq['question']}")
            print(f"Correct Answer: {mcq['correct_answer']}")
            print(f"Distractors: {', '.join(mcq['distractors'])}")
        
    except Exception as e:
        print(f"❌ Complete MCQ generation error: {e}")
    
    print("\n🎉 Component testing completed!")

if __name__ == "__main__":
    asyncio.run(test_individual_components())
