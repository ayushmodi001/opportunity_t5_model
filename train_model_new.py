import json
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCQDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        
        input_text = f"context: {context[:256]} question:"
        target_text = question
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length//2,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def load_dataset_from_json():
    data_path = Path('data/combined_dataset.json')
    if not data_path.exists():
        raise FileNotFoundError("Dataset not found. Run prepare_dataset.py first.")
    
    with open(data_path) as f:
        data = json.load(f)
    
    np.random.shuffle(data)
    split = int(0.9 * len(data))
    return data[:split], data[split:]

def train_model():
    try:
        # Use local model
        model_path = "./t5_model"
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        # Load and prepare data
        train_data, val_data = load_dataset_from_json()
        train_dataset = MCQDataset(train_data, tokenizer)
        val_dataset = MCQDataset(val_data, tokenizer)
        
        # Memory optimized training args
        training_args = TrainingArguments(
            output_dir="./mcq_model",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            do_eval=True,
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            optim="adamw_torch",
            gradient_checkpointing=True,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save optimized model
        final_model_path = "./mcq_model/final"
        model.save_pretrained(
            final_model_path,
            max_shard_size="200MB"
        )
        tokenizer.save_pretrained(final_model_path)
        
        # Export to ONNX
        logger.info("Converting to ONNX format...")
        dummy_input = tokenizer("convert question:", return_tensors="pt")
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            f"{final_model_path}/model.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            },
            opset_version=12
        )
        
        logger.info("Training completed and model saved!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
