#!/usr/bin/env python3

import os
import logging
import tweepy
from datetime import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Load environment variables
load_dotenv()

# Twitter API credentials
TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

def test_twitter_api():
    """Test Twitter API connectivity and posting capabilities"""
    try:
        # Set up Twitter API v2 client
        client = tweepy.Client(
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN, 
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
        )
        
        # Create a test tweet
        test_message = f"Testing API access at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} #test"
        logger.info(f"Attempting to post: {test_message}")
        
        # Post the tweet
        response = client.create_tweet(text=test_message)
        
        # Log the results
        if response and response.data:
            logger.info(f"Tweet successfully posted with ID: {response.data['id']}")
            logger.info(f"Full response data: {response.data}")
            return True
        else:
            logger.error("Failed to post tweet, no response data")
            return False
            
    except Exception as e:
        logger.error(f"Error testing Twitter API: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting Twitter API test")
    result = test_twitter_api()
    logger.info(f"Test result: {'Success' if result else 'Failed'}")