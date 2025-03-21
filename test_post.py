#!/usr/bin/env python3
import os
import tweepy
from dotenv import load_dotenv

load_dotenv()

# Twitter API credentials
TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# Twitter API v1 for tweeting
auth = tweepy.OAuth1UserHandler(
    TWITTER_CONSUMER_KEY,
    TWITTER_CONSUMER_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_TOKEN_SECRET
)
api = tweepy.API(auth)

# Attempt to post a test tweet
try:
    api.update_status("This is a test tweet from my crypto bot system. #testing " + str(os.urandom(4).hex()))
    print("Test tweet posted successfully")
except Exception as e:
    print(f"Error posting test tweet: {str(e)}")
