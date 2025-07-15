#!/usr/bin/env python3
"""
Install core dependencies only - avoiding build issues
"""
import subprocess
import sys

def install_core_only():
    """Install only the essential packages that don't require compilation"""
    print("🚀 Installing Core Dependencies Only")
    print("=" * 40)
    
    # Essential packages that should work on Python 3.13
    essential_packages = [
        "fastapi",
        "uvicorn",
        "python-dotenv", 
        "requests",
        "mistralai",
    ]
    
    for package in essential_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    # Try to install transformers with --no-build-isolation
    print("\nTrying to install transformers...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "transformers", "--no-build-isolation"
        ], check=True)
        print("✅ transformers installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️ transformers installation failed - will use mock generation")
    
    print("\n🎉 Core installation completed!")
    print("You can now test with: python validate_system.py")

if __name__ == "__main__":
    install_core_only()
