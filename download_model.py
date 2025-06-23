from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

def download_model():
    print("Downloading T5-small model...")
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Save locally
    output_dir = "./t5_model"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving model locally...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model downloaded and saved successfully!")

if __name__ == "__main__":
    download_model()
