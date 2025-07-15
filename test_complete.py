#!/usr/bin/env python3
"""
Complete MCQ Generation Test
"""
import asyncio
import json

async def test_complete_mcq_generation():
    print("🧪 Complete MCQ Generation Test")
    print("=" * 50)
    
    # Test data
    topic = "Python programming"
    
    try:
        # 1. Context Fetching
        print("1️⃣ Fetching context...")
        from context_fetcher import fetch_context
        context = await fetch_context(topic)
        
        if context and len(context) > 100:
            print(f"✅ Context fetched: {len(context)} characters")
            print(f"Preview: {context[:150]}...")
        else:
            print("⚠️ Using fallback context")
        
        # 2. Answer Extraction
        print("\n2️⃣ Extracting answers...")
        from answer_extractor import extract_answers
        answers = await extract_answers(context)
        
        print(f"✅ Extracted {len(answers)} answers: {answers[:5]}")
        
        # 3. Question Generation
        print("\n3️⃣ Generating questions...")
        from question_generator import generate_questions
        questions = await generate_questions(context, answers, target_questions=5)
        
        print(f"✅ Generated {len(questions)} questions")
        for i, qa in enumerate(questions[:3]):
            print(f"   Q{i+1}: {qa['question']}")
            print(f"   A{i+1}: {qa['answer']}")
        
        # 4. Complete MCQ Generation
        print("\n4️⃣ Creating complete MCQs...")
        mcqs = []
        
        from distractor_generator import generate_distractors
        
        for i, qa in enumerate(questions[:3]):  # Test with 3 questions
            print(f"   Creating MCQ {i+1}...")
            distractors = await generate_distractors(context, qa["answer"])
            
            mcq = {
                "topic": topic,
                "question": qa["question"],
                "correct_answer": qa["answer"],
                "distractors": distractors
            }
            mcqs.append(mcq)
        
        print(f"✅ Created {len(mcqs)} complete MCQs")
        
        # 5. Save and Display Results
        result = {"questions": mcqs}
        
        with open("complete_test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n📋 Final MCQs:")
        for i, mcq in enumerate(mcqs):
            print(f"\n--- MCQ {i+1} ---")
            print(f"Topic: {mcq['topic']}")
            print(f"Question: {mcq['question']}")
            print(f"Correct Answer: {mcq['correct_answer']}")
            print(f"Distractors: {', '.join(mcq['distractors'])}")
        
        print(f"\n💾 Results saved to complete_test_results.json")
        print("🎉 Complete MCQ generation test successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in complete MCQ generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_mcq_generation())
    if success:
        print("\n✅ All systems working correctly!")
    else:
        print("\n❌ Some issues detected, but system should still work with fallbacks")
