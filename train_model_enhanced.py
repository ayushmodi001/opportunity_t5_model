import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import numpy as np
import time

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
          # Optimized training args for GTX 1650 (4GB VRAM)
        training_args = TrainingArguments(
            output_dir="./mcq_model",
            num_train_epochs=3,
            per_device_train_batch_size=2,  # Reduced batch size
            per_device_eval_batch_size=2,   # Reduced eval batch size
            gradient_accumulation_steps=8,   # Increased for effective batch size of 16
            learning_rate=2e-5,             # Slightly lower learning rate for stability
            warmup_ratio=0.1,              # Add warmup steps
            do_eval=True,
            eval_strategy="steps",         # Evaluate during training
            eval_steps=100,                # More frequent evaluation
            logging_steps=50,              # More frequent logging
            save_steps=500,
            save_total_limit=2,            # Keep only last 2 checkpoints
            fp16=True,                     # Enable mixed precision training
            gradient_checkpointing=True,   # Memory optimization
            weight_decay=0.01,            # L2 regularization
            max_grad_norm=0.5,            # Clip gradients at 0.5
            optim="adamw_torch",
            dataloader_num_workers=2,     # Parallel data loading
            remove_unused_columns=True,   # Memory optimization
            load_best_model_at_end=True,  # Save best model
        )
        
        # Enable torch.cuda memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,    
            eval_dataset=val_dataset
        )
          # Train
        print("\n" + "="*50)
        print("📚 Starting Model Training...")
        print("="*50)
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Print training summary
        print("\n" + "="*50)
        print("🎓 Training Complete!")
        print("="*50)
        print(f"⏱️  Total Training Time: {training_time/3600:.2f} hours")
        print(f"💾 Saving Model...")
        
        # Save the final model
        final_model_path = "./mcq_model/final"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print("✅ Model Saved Successfully!")
        
        # Convert to ONNX
        print("\n" + "="*50)
        print("🔄 Converting Model to ONNX Format...")
        print("="*50)
        convert_to_onnx(final_model_path)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def convert_to_onnx(model_path: str):
    """Convert the trained model to ONNX format with proper device handling."""
    try:
        import torch.onnx
        
        print("📥 Loading model for conversion...")
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        print("🔄 Preparing model for export...")
        # Move model to CPU for conversion to avoid device mismatch
        model = model.cpu()
        model.eval()
        
        # Create dummy inputs for encoder and decoder
        encoder_input = tokenizer(
            "context: This is a sample context question:",
            return_tensors="pt",
            max_length=128,
            padding='max_length',
            truncation=True
        )
        
        decoder_input = tokenizer(
            "sample output",
            return_tensors="pt",
            max_length=64,
            padding='max_length',
            truncation=True
        )
        
        # Prepare inputs
        input_ids = encoder_input['input_ids']
        attention_mask = encoder_input['attention_mask']
        decoder_input_ids = decoder_input['input_ids']
        decoder_attention_mask = decoder_input['attention_mask']
        
        # Create a wrapper class for ONNX export
        class T5EncoderWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                
            def forward(self, input_ids, attention_mask):
                return self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        class T5DecoderWrapper(torch.nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder
                
            def forward(self, input_ids, attention_mask, encoder_hidden_states):
                return self.decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states
                )[0]
          # Export encoder
        print("🔄 Exporting encoder model...")
        encoder_wrapper = T5EncoderWrapper(model.encoder)
        encoder_path = f"{model_path}/encoder.onnx"
        torch.onnx.export(
            encoder_wrapper,
            (input_ids, attention_mask),
            encoder_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['encoder_hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'encoder_hidden_states': {0: 'batch_size', 1: 'sequence_length', 2: 'hidden_size'}
            },
            do_constant_folding=True,
            opset_version=12
        )
        
        # Get encoder output for decoder export
        with torch.no_grad():
            encoder_hidden_states = encoder_wrapper(input_ids, attention_mask)
          # Export decoder
        print("🔄 Exporting decoder model...")
        decoder_wrapper = T5DecoderWrapper(model.decoder)
        decoder_path = f"{model_path}/decoder.onnx"
        torch.onnx.export(
            decoder_wrapper,
            (decoder_input_ids, decoder_attention_mask, encoder_hidden_states),
            decoder_path,
            input_names=['input_ids', 'attention_mask', 'encoder_hidden_states'],
            output_names=['decoder_output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'target_sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'target_sequence_length'},
                'encoder_hidden_states': {0: 'batch_size', 1: 'sequence_length', 2: 'hidden_size'},
                'decoder_output': {0: 'batch_size', 1: 'target_sequence_length', 2: 'vocab_size'}
            },
            do_constant_folding=True,
            opset_version=12        )
        print("\n" + "="*50)
        print("✨ Model Export Complete!")
        print("="*50)
        print(f"📁 Encoder saved to: {encoder_path}")
        print(f"📁 Decoder saved to: {decoder_path}")
        print("🎉 All done! Your model is ready for deployment!")
        
    except Exception as e:
        print("\n❌ Error during model conversion:")
        print(f"   {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
