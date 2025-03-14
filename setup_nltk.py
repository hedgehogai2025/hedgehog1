#!/usr/bin/env python3
import nltk
import os
import sys

def main():
    print("Setting up NLTK data...")
    
    # Create directory for NLTK data
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Tell NLTK where to find the data
    nltk.data.path.append(nltk_data_dir)
    
    # Download required data packages
    for package in ['punkt', 'vader_lexicon', 'stopwords']:
        print(f"Downloading {package}...")
        nltk.download(package, download_dir=nltk_data_dir)
    
    print("NLTK data download complete.")
    print(f"NLTK data directory: {nltk_data_dir}")
    print("To use this data, set the NLTK_DATA environment variable:")
    print(f"export NLTK_DATA={nltk_data_dir}")

if __name__ == "__main__":
    main()