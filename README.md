# MCQ Generator

A T5-based Multiple Choice Question (MCQ) generator that creates high-quality questions for various topics including computer science skills, academic courses, and career paths.

## Features

- Generates context-aware MCQs using T5 transformer model
- Utilizes GPU acceleration (GTX 1650) for faster processing
- Fetches relevant content from multiple sources
- Generates meaningful distractors using WordNet
- Modular and scalable architecture
- Efficient and fast generation for real-time use

## Requirements

- Python 3.8+
- CUDA-compatible GPU (tested with GTX 1650)
- See requirements.txt for Python packages

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mcq
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Generate MCQs using the command line interface:

```bash
python src/main.py --topic "javascript programming" --num_questions 10 --output mcqs.json
```

Parameters:
- `--topic`: The topic to generate MCQs for (e.g., "javascript", "btech in computer science", "business analyst")
- `--num_questions`: Number of MCQs to generate (default: 10)
- `--output`: Output file path for saving MCQs (default: mcqs.json)

## Project Structure

```
mcq/
├── src/
│   ├── main.py              # Main entry point
│   ├── mcq_generator.py     # MCQ generation using T5
│   └── utils/
│       ├── text_processing.py  # Text processing utilities
│       └── content_fetcher.py  # Content fetching from various sources
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Future Improvements

1. API Integration
   - REST API endpoint for MCQ generation
   - Authentication and rate limiting

2. Cloud Deployment
   - Containerization using Docker
   - Cloud deployment scripts
   - Cost optimization strategies

3. Enhanced Features
   - More content sources
   - Improved distractor generation
   - Question difficulty levels
   - Topic-specific templates

## Note

This is a proof-of-concept implementation. The quality of generated MCQs depends on:
- Quality of input content
- T5 model fine-tuning
- Keyword extraction accuracy
- Distractor generation method

For production use, consider:
- Fine-tuning the T5 model on domain-specific data
- Implementing caching for frequently requested topics
- Adding validation for generated questions
- Implementing proper error handling and logging
