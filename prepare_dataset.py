import os
import json
import logging
from datasets import load_dataset
from typing import Dict, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sciq_dataset() -> List[Dict]:
    """Load and process SciQ dataset"""
    try:
        dataset = load_dataset("sciq")
        processed = []
        
        for split in ['train', 'validation', 'test']:
            for item in dataset[split]:
                processed.append({
                    'context': item['support'],
                    'question': item['question'],
                    'correct_answer': item['correct_answer'],
                    'distractors': [item['distractor1'], item['distractor2'], item['distractor3']],
                    'source': 'sciq'
                })
        
        return processed
    except Exception as e:
        logger.error(f"Error loading SciQ dataset: {str(e)}")
        return []

def get_mcql_dataset() -> List[Dict]:
    """Load and process MMLU computer science dataset"""
    try:
        # Using standard MMLU dataset
        dataset = load_dataset("cais/mmlu", "college_computer_science", trust_remote_code=True)
        processed = []
        
        for split in ['auxiliary_train', 'test', 'validation']:  # MMLU has multiple good sets
            for item in dataset[split]:
                if not (item.get('input') and item.get('target') is not None and 
                        isinstance(item.get('target'), int) and
                        isinstance(item.get('choices'), list)):
                    continue
                
                question = item['input']
                options = item['choices']
                if not options or len(options) < 4:
                    continue
                
                correct_answer = options[item['target']]
                distractors = [opt for i, opt in enumerate(options) if i != item['target']][:3]
                
                if len(distractors) < 3:
                    continue
                
                processed.append({
                    'context': '',  # MMLU doesn't provide context
                    'question': question.strip(),
                    'correct_answer': correct_answer.strip(),
                    'distractors': [d.strip() for d in distractors],
                    'source': 'mmlu_cs'
                })
        
        return processed
    except Exception as e:
        logger.error(f"Error loading MMLU dataset: {str(e)}")
        return []

def get_backup_mcq_dataset() -> List[Dict]:
    """Load and process backup MCQ dataset if primary source fails"""
    try:
        # Using Stanford CS MCQ dataset
        dataset = load_dataset("stem_mcq")
        processed = []
        
        if 'train' in dataset:
            for item in dataset['train']:
                question = item.get('question', '')
                options = item.get('options', [])
                correct_idx = item.get('answer_index', -1)
                
                if not (question and options and 
                       isinstance(correct_idx, int) and correct_idx >= 0 and
                       len(options) >= 4):
                    continue
                
                correct_answer = options[correct_idx]
                distractors = [opt for i, opt in enumerate(options) if i != correct_idx][:3]
                
                if len(distractors) < 3:
                    continue
                
                processed.append({
                    'context': item.get('context', ''),
                    'question': question.strip(),
                    'correct_answer': correct_answer.strip(),
                    'distractors': [d.strip() for d in distractors],
                    'source': 'stem_mcq'
                })
        
        return processed
    except Exception as e:
        logger.error(f"Error loading backup dataset: {str(e)}")
        return []

def get_squad_tech_mcqs() -> List[Dict]:
    """Convert relevant technical questions to MCQ format"""
    try:
        # Using SQUAD subset focused on technical content
        dataset = load_dataset("squad_v2")
        processed = []
        
        tech_keywords = ['algorithm', 'computer', 'software', 'hardware', 'programming',
                        'network', 'database', 'system', 'protocol', 'technology',
                        'code', 'application', 'interface', 'server', 'client']
        
        if 'train' in dataset:
            for item in dataset['train']:
                # Check if question is technical
                context = item.get('context', '')
                question = item.get('question', '')
                answers = item.get('answers', {})
                
                if not (context and question and answers and 
                       any(keyword in question.lower() or keyword in context.lower() 
                           for keyword in tech_keywords)):
                    continue
                
                if not answers.get('text'):
                    continue
                
                processed.append({
                    'context': context,
                    'question': question,
                    'correct_answer': answers['text'][0],
                    'source': 'squad_tech'
                })
        
        return processed
    except Exception as e:
        logger.error(f"Error loading technical QA dataset: {str(e)}")
        return []

def combine_datasets() -> List[Dict]:
    """Combine all datasets and format for training"""
    all_data = []
    
    # Load primary datasets
    sciq_data = get_sciq_dataset()
    mcq_data = get_mcql_dataset()
    tech_data = get_squad_tech_mcqs()
    
    # If primary MCQ dataset is empty, try backup
    if not mcq_data:
        logger.info("Primary MCQ dataset failed to load, trying backup dataset...")
        mcq_data = get_backup_mcq_dataset()
    
    logger.info(f"Loaded {len(sciq_data)} examples from SciQ")
    logger.info(f"Loaded {len(mcq_data)} examples from MCQ dataset")
    logger.info(f"Loaded {len(tech_data)} examples from technical QA dataset")
    
    all_data.extend(sciq_data)
    all_data.extend(mcq_data)
    all_data.extend(tech_data)
    
    # Save combined dataset
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'combined_dataset.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    logger.info(f"Combined dataset saved with {len(all_data)} examples")
    return all_data

if __name__ == "__main__":
    combine_datasets()
