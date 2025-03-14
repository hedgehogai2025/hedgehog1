#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data
mkdir -p charts
mkdir -p templates
mkdir -p logs

# Copy .env-example to .env if not exists
if [ ! -f .env ]; then
    cp .env-example .env
    echo "Please edit .env file with your API keys"
fi

# Make main script executable
chmod +x src/main.py

echo "Setup complete!"
