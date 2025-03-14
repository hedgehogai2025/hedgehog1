#!/usr/bin/env python3
import time
import os

with open('/home/ubuntu/crypto_analysis_bot/logs/test.log', 'a') as f:
    f.write(f"Script started at {time.ctime()}\n")
    f.write(f"Current directory: {os.getcwd()}\n")

print("Test script executed successfully")