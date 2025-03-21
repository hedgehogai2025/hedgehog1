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

# Print masked credentials to verify they are loaded
print(f"Consumer Key: {TWITTER_CONSUMER_KEY[:4]}...{TWITTER_CONSUMER_KEY[-4:]}")
print(f"Consumer Secret: {TWITTER_CONSUMER_SECRET[:4]}...{TWITTER_CONSUMER_SECRET[-4:]}")
print(f"Access Token: {TWITTER_ACCESS_TOKEN[:4]}...{TWITTER_ACCESS_TOKEN[-4:]}")
print(f"Access Secret: {TWITTER_ACCESS_TOKEN_SECRET[:4]}...{TWITTER_ACCESS_TOKEN_SECRET[-4:]}")

# Test v1.1 API
try:
    auth_v1 = tweepy.OAuth1UserHandler(
        TWITTER_CONSUMER_KEY,
        TWITTER_CONSUMER_SECRET,
        TWITTER_ACCESS_TOKEN,
        TWITTER_ACCESS_TOKEN_SECRET
    )
    api_v1 = tweepy.API(auth_v1)
    
    # Verify credentials
    user = api_v1.verify_credentials()
    print(f"Successfully authenticated as: {user.screen_name}")
    
    # Attempt to post a tweet
    test_tweet = api_v1.update_status("This is a test tweet from my crypto bot diagnostic tool. " + str(os.urandom(4).hex()))
    print(f"Successfully posted tweet! ID: {test_tweet.id}")
    
except Exception as e:
    print(f"Error with v1.1 API: {str(e)}")

# Test v2 API
try:
    client_v2 = tweepy.Client(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
    )
    
    # Attempt to post a tweet with v2 API
    test_tweet_v2 = client_v2.create_tweet(text="This is a v2 API test tweet from my crypto bot diagnostic tool. " + str(os.urandom(4).hex()))
    print(f"Successfully posted tweet via v2 API! ID: {test_tweet_v2.data['id']}")
    
except Exception as e:
    print(f"Error with v2 API: {str(e)}")