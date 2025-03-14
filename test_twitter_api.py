import os
from dotenv import load_dotenv
import tweepy
import sys

def main():
    load_dotenv()
    
    print("Testing Twitter API access...")
    
    # API credentials
    consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
    consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
    access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
    # Test v2 API
    print("\nTesting API v2...")
    try:
        client = tweepy.Client(
            consumer_key=consumer_key, 
            consumer_secret=consumer_secret,
            access_token=access_token, 
            access_token_secret=access_token_secret
        )
        
        # Get account info
        me = client.get_me()
        print(f"Connected as: @{me.data.username} (ID: {me.data.id})")
        
        # Test posting if argument is provided
        if len(sys.argv) > 1 and sys.argv[1] == '--post':
            print("Attempting to post a test tweet...")
            tweet = client.create_tweet(text="This is a test tweet from my crypto analysis bot. " + 
                                         "If you see this, the API is working correctly! " +
                                         str(datetime.now()))
            print(f"Tweet posted successfully! ID: {tweet.data['id']}")
    except Exception as e:
        print(f"Error with v2 API: {str(e)}")
    
    print("\nTwitter API test complete")

if __name__ == "__main__":
    from datetime import datetime
    main()