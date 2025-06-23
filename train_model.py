import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import nltk
from tqdm import tqdm
import os
import json

class MCQDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['support']  # The supporting text/context
        question = item['question']
        correct_answer = item['correct_answer']
        
        # Format input text
        input_text = f"context: {context} answer: {correct_answer} </s>"
        target_text = f"question: {question} </s>"

        # Tokenize inputs and targets
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def prepare_data():
    """Load and prepare SciQ dataset"""
    dataset = load_dataset("sciq")
    train_data = dataset['train']
    val_data = dataset['validation']
    
    return train_data, val_data

def train_model():
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "t5-small"  # Using smaller model for efficiency
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Prepare datasets
    train_data, val_data = prepare_data()
    
    # Create custom datasets
    train_dataset = MCQDataset(train_data, tokenizer)
    val_dataset = MCQDataset(val_data, tokenizer)    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./mcq_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    print("Starting model training...")
    trainer.train()

    # Save the fine-tuned model
    model_save_path = "./mcq_model/final"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
