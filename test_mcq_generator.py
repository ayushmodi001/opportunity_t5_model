import os
import json
import asyncio
from typing import Dict, List
import statistics
from generate_mcq_new import MCQGenerator

class MCQTester:
    def __init__(self):
        self.generator = MCQGenerator()
        self.results: List[Dict] = []
        
    async def test_topic(self, topic: str) -> Dict:
        """Test MCQ generation for a single topic"""
        print(f"\nTesting MCQ generation for topic: {topic}")
        result = await self.generator.test_mcq_quality(topic)
        self.results.append(result)
        
        if result["status"] == "success":
            print("✅ MCQ generation successful")
            print(f"Question: {result['question']}")
            print(f"Correct Answer: {result['correct_answer']}")
            print("Distractors:")
            for d in result['distractors']:
                print(f"- {d}")
            print("\nQuality Metrics:")
            self._print_quality_metrics(result['quality_metrics'])
        else:
            print("❌ MCQ generation failed:")
            print(result["message"])
            
        return result
        
    def _print_quality_metrics(self, metrics: Dict):
        """Print formatted quality metrics"""
        print("Question Metrics:")
        print(f"- Length: {metrics['question']['length']} words")
        print(f"- Has topic context: {'✅' if metrics['question']['has_context'] else '❌'}")
        print(f"- Ends with question mark: {'✅' if metrics['question']['ends_with_question'] else '❌'}")
        
        print("\nOption Metrics:")
        lengths = metrics['options']['lengths']
        similarities = metrics['options']['similarity_ratios']
        print(f"- Option lengths: {lengths}")
        print(f"- Average length: {statistics.mean(lengths):.1f} words")
        print(f"- Length std dev: {statistics.stdev(lengths):.1f}")
        print(f"- Average similarity: {statistics.mean(similarities):.2f}")
        print(f"- Max similarity: {max(similarities):.2f}")
        
    def save_results(self, filename: str = "mcq_test_results.json"):
        """Save test results to a JSON file"""
        stats = self._calculate_aggregate_stats()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "total_tests": len(self.results),
                "successful_tests": stats["successful_tests"],
                "failed_tests": stats["failed_tests"],
                "aggregated_metrics": stats["metrics"],
                "results": self.results
            }, f, indent=2)
        print(f"\nResults saved to {filename}")
        
    def _calculate_aggregate_stats(self) -> Dict:
        """Calculate aggregated statistics across all tests"""
        successful_tests = [r for r in self.results if r["status"] == "success"]
        failed_tests = [r for r in self.results if r["status"] == "error"]
        
        if not successful_tests:
            return {
                "successful_tests": 0,
                "failed_tests": len(failed_tests),
                "metrics": {}
            }
            
        # Aggregate question metrics
        question_lengths = []
        has_context_count = 0
        question_mark_count = 0
        
        # Aggregate option metrics
        option_lengths = []
        similarity_ratios = []
        
        for test in successful_tests:
            metrics = test["quality_metrics"]
            
            # Question metrics
            question_lengths.append(metrics["question"]["length"])
            if metrics["question"]["has_context"]:
                has_context_count += 1
            if metrics["question"]["ends_with_question"]:
                question_mark_count += 1
                
            # Option metrics
            option_lengths.extend(metrics["options"]["lengths"])
            similarity_ratios.extend(metrics["options"]["similarity_ratios"])
            
        return {
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "metrics": {
                "question": {
                    "avg_length": statistics.mean(question_lengths),
                    "length_std_dev": statistics.stdev(question_lengths),
                    "context_rate": has_context_count / len(successful_tests),
                    "question_mark_rate": question_mark_count / len(successful_tests)
                },
                "options": {
                    "avg_length": statistics.mean(option_lengths),
                    "length_std_dev": statistics.stdev(option_lengths),
                    "avg_similarity": statistics.mean(similarity_ratios),
                    "max_similarity": max(similarity_ratios)
                }
            }
        }
        
    def print_summary(self):
        """Print a summary of test results"""
        stats = self._calculate_aggregate_stats()
        total = stats["successful_tests"] + stats["failed_tests"]
        
        print("\nTest Summary:")
        print(f"Total tests: {total}")
        print(f"Successful: {stats['successful_tests']} ({(stats['successful_tests']/total)*100:.1f}%)")
        print(f"Failed: {stats['failed_tests']} ({(stats['failed_tests']/total)*100:.1f}%)")
        
        if stats["successful_tests"] > 0:
            metrics = stats["metrics"]
            print("\nAggregated Metrics:")
            print("\nQuestion Quality:")
            print(f"- Average length: {metrics['question']['avg_length']:.1f} words")
            print(f"- Length std dev: {metrics['question']['length_std_dev']:.1f}")
            print(f"- Topic context rate: {metrics['question']['context_rate']*100:.1f}%")
            print(f"- Question mark rate: {metrics['question']['question_mark_rate']*100:.1f}%")
            
            print("\nOption Quality:")
            print(f"- Average length: {metrics['options']['avg_length']:.1f} words")
            print(f"- Length std dev: {metrics['options']['length_std_dev']:.1f}")
            print(f"- Average similarity: {metrics['options']['avg_similarity']:.2f}")
            print(f"- Max similarity: {metrics['options']['max_similarity']:.2f}")

async def main():
    tester = MCQTester()
    
    # Test topics covering different domains
    test_topics = [
        "Machine Learning",
        "Data Structures",
        "Database Management",
        "Software Engineering",
        "Computer Networks",
        "Operating Systems",
        "Web Development",
        "Artificial Intelligence"
    ]
    
    # Run tests with exponential backoff for retries
    for topic in test_topics:
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                await tester.test_topic(topic)
                print("-" * 80)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"❌ Failed to test topic '{topic}' after {max_retries} attempts: {str(e)}")
                else:
                    delay = retry_delay * (2 ** attempt)
                    print(f"⚠️ Attempt {attempt + 1} failed, retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
    
    # Print and save results
    tester.print_summary()
    tester.save_results()

if __name__ == "__main__":
    asyncio.run(main())