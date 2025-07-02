#!/usr/bin/env python3
"""
Setup script for Resume Summarizer and Recommendation System
This script helps users set up the environment and download required resources.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Successfully installed all requirements!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ Successfully downloaded NLTK data!")
    except ImportError:
        print("❌ NLTK not installed. Please install requirements first.")
        return False
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def check_model_files():
    """Check if model files exist"""
    model_files = ['resume_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl']
    missing_files = []
    
    for file in model_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing model files: {', '.join(missing_files)}")
        print("🔧 Please run the Jupyter notebook 'resume_recommendation.ipynb' to train and save the models.")
        return False
    else:
        print("✅ All model files are present!")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up Resume Summarizer and Recommendation System...")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation.")
        return
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Setup failed during NLTK data download.")
        return
    
    # Check model files
    model_files_exist = check_model_files()
    
    print("\n" + "=" * 60)
    if model_files_exist:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run the web app: streamlit run 'Resume recommendation and summarize app.py'")
        print("2. Open your browser and navigate to the provided URL")
        print("3. Upload a resume and get recommendations!")
    else:
        print("⚠️  Setup partially completed!")
        print("\n📋 Next steps:")
        print("1. Open and run 'resume_recommendation.ipynb' to train the models")
        print("2. After training, run: streamlit run 'Resume recommendation and summarize app.py'")
        print("3. Upload a resume and get recommendations!")

if __name__ == "__main__":
    main()
