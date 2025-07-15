import asyncio
import json
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_mcq_generation():
    """Test the complete MCQ generation pipeline"""
    try:
        from context_fetcher import fetch_context
        from answer_extractor import extract_answers
        from question_generator import generate_questions
        from distractor_generator import generate_distractors
        
        print("🚀 Testing MCQ Generation Pipeline")
        print("=" * 50)
        
        # Test topic
        topic = "Python programming"
        
        print(f"📚 Topic: {topic}")
        
        # 1. Fetch context
        print("\n1️⃣ Fetching context...")
        try:
            context = await fetch_context(topic)
            print(f"✅ Context fetched ({len(context.split())} words)")
            print(f"Sample: {context[:200]}...")
        except Exception as e:
            print(f"❌ Error fetching context: {e}")
            return
        
        # 2. Extract answers
        print("\n2️⃣ Extracting answers...")
        try:
            answers = await extract_answers(context)
            print(f"✅ Extracted {len(answers)} answers: {answers[:5]}...")
        except Exception as e:
            print(f"❌ Error extracting answers: {e}")
            return
        
        if not answers:
            print("❌ No answers extracted, cannot continue")
            return
        
        # 3. Generate questions
        print("\n3️⃣ Generating questions...")
        try:
            questions = await generate_questions(context, answers, target_questions=5)
            print(f"✅ Generated {len(questions)} questions")
            for i, q in enumerate(questions[:3]):
                print(f"   Q{i+1}: {q['question']}")
                print(f"       A: {q['answer']}")
        except Exception as e:
            print(f"❌ Error generating questions: {e}")
            return
        
        # 4. Generate distractors for first question
        if questions:
            print("\n4️⃣ Generating distractors...")
            try:
                first_q = questions[0]
                distractors = await generate_distractors(context, first_q["answer"])
                print(f"✅ Generated distractors for '{first_q['answer']}':")
                for i, distractor in enumerate(distractors):
                    print(f"   D{i+1}: {distractor}")
            except Exception as e:
                print(f"❌ Error generating distractors: {e}")
        
        print("\n🎉 Test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcq_generation())
