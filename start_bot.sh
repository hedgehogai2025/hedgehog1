#!/bin/bash

# Navigate to the project directory
cd /home/ubuntu/crypto_analysis_bot

# Activate the virtual environment
source /home/ubuntu/crypto_analysis_bot/venv/bin/activate

# Start the bot with error output
python3 /home/ubuntu/crypto_analysis_bot/src/main.py >> /home/ubuntu/crypto_analysis_bot/logs/runtime.log 2>> /home/ubuntu/crypto_analysis_bot/logs/error.log