#!/usr/bin/env python3
"""
Setup script to install dependencies safely for Python 3.13
"""
import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def install_packages():
    """Install packages one by one to avoid conflicts"""
    print("🚀 Setting up MCQ Generation System")
    print("=" * 40)
    
    # Core packages first
    core_packages = [
        "pip --upgrade",
        "python-dotenv",
        "requests",
        "pydantic>=2.0.0",
        "fastapi",
        "uvicorn[standard]",
    ]
    
    print("1️⃣ Installing core packages...")
    for package in core_packages:
        if not run_command(f"pip install {package}"):
            print(f"⚠️ Warning: Failed to install {package}")
    
    # AI/ML packages
    ai_packages = [
        "transformers",
        "torch --index-url https://download.pytorch.org/whl/cpu",  # CPU version for compatibility
        "mistralai",
    ]
    
    print("\n2️⃣ Installing AI/ML packages...")
    for package in ai_packages:
        if not run_command(f"pip install {package}"):
            print(f"⚠️ Warning: Failed to install {package}")
    
    # NLP packages
    nlp_packages = [
        "spacy",
        "yake",
    ]
    
    print("\n3️⃣ Installing NLP packages...")
    for package in nlp_packages:
        if not run_command(f"pip install {package}"):
            print(f"⚠️ Warning: Failed to install {package}")
    
    # Download spaCy model
    print("\n4️⃣ Downloading spaCy model...")
    if not run_command("python -m spacy download en_core_web_sm"):
        print("⚠️ Warning: Failed to download spaCy model - will use fallback")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Check if all imports work: python validate_system.py")
    print("2. Start the server: python -m uvicorn main:app --reload")

def create_minimal_requirements():
    """Create a minimal requirements file for manual installation"""
    minimal_reqs = """# Minimal requirements - install manually if automated setup fails
fastapi
uvicorn
python-dotenv
requests
transformers
torch
mistralai
yake
spacy
"""
    
    with open("requirements_minimal.txt", "w") as f:
        f.write(minimal_reqs)
    
    print("📝 Created requirements_minimal.txt for manual installation")

if __name__ == "__main__":
    try:
        install_packages()
    except KeyboardInterrupt:
        print("\n⚠️ Installation interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\n💡 Try manual installation:")
        create_minimal_requirements()
        print("pip install -r requirements_minimal.txt")
