import os
import tweepy
from dotenv import load_dotenv
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/twitter_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TwitterClient:
    def __init__(self):
        self.consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
        self.consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Use both v1 and v2 clients
        self.api_v1 = self._initialize_api_v1()
        self.client_v2 = self._initialize_client_v2()
        
    def _initialize_api_v1(self):
        """Initialize and return the Twitter API v1.1 client."""
        try:
            auth = tweepy.OAuth1UserHandler(
                self.consumer_key, 
                self.consumer_secret,
                self.access_token, 
                self.access_token_secret
            )
            api = tweepy.API(auth, wait_on_rate_limit=True)
            logger.info("Twitter API v1.1 initialized")
            return api
        except Exception as e:
            logger.error(f"Error initializing Twitter API v1.1: {str(e)}")
            return None
            
    def _initialize_client_v2(self):
        """Initialize and return the Twitter API v2 client."""
        try:
            client = tweepy.Client(
                consumer_key=self.consumer_key, 
                consumer_secret=self.consumer_secret,
                access_token=self.access_token, 
                access_token_secret=self.access_token_secret,
                wait_on_rate_limit=True
            )
            logger.info("Twitter API v2 initialized")
            return client
        except Exception as e:
            logger.error(f"Error initializing Twitter API v2: {str(e)}")
            return None
    
    def get_kol_tweets(self, kol_usernames, count=100):
        """Fetch recent tweets from a list of KOL usernames."""
        all_tweets = []
        
        for username in kol_usernames:
            try:
                if self.api_v1:
                    tweets = self.api_v1.user_timeline(screen_name=username, count=count, tweet_mode='extended')
                    logger.info(f"Fetched {len(tweets)} tweets from {username}")
                    
                    for tweet in tweets:
                        tweet_data = {
                            'id': tweet.id,
                            'username': username,
                            'text': tweet.full_text,
                            'created_at': tweet.created_at,
                            'retweet_count': tweet.retweet_count,
                            'favorite_count': tweet.favorite_count
                        }
                        all_tweets.append(tweet_data)
                elif self.client_v2:
                    # First get user ID from username
                    user = self.client_v2.get_user(username=username)
                    if user.data:
                        user_id = user.data.id
                        tweets = self.client_v2.get_users_tweets(
                            id=user_id, 
                            max_results=count,
                            tweet_fields=['created_at', 'public_metrics']
                        )
                        
                        if tweets.data:
                            logger.info(f"Fetched {len(tweets.data)} tweets from {username}")
                            
                            for tweet in tweets.data:
                                tweet_data = {
                                    'id': tweet.id,
                                    'username': username,
                                    'text': tweet.text,
                                    'created_at': tweet.created_at,
                                    'retweet_count': tweet.public_metrics.get('retweet_count', 0),
                                    'favorite_count': tweet.public_metrics.get('like_count', 0)
                                }
                                all_tweets.append(tweet_data)
                    else:
                        logger.error(f"Could not find user with username {username}")
                else:
                    logger.error("No Twitter API client available")
                    
            except Exception as e:
                logger.error(f"Error fetching tweets from {username}: {str(e)}")
            
            # Sleep to avoid rate limiting
            time.sleep(2)
            
        return all_tweets
    
    def post_tweet(self, text):
        """Post a tweet with the given text."""
        try:
            # Try v2 API first
            if self.client_v2:
                response = self.client_v2.create_tweet(text=text)
                if response and hasattr(response, 'data'):
                    tweet_id = response.data.get('id')
                    logger.info(f"Posted tweet via v2 API (ID: {tweet_id}): {text[:50]}...")
                    return response
                else:
                    logger.error(f"Unexpected response from v2 API: {response}")
            
            # Fall back to v1 if v2 fails or isn't available
            if self.api_v1 and (not self.client_v2 or not response):
                tweet = self.api_v1.update_status(text)
                logger.info(f"Posted tweet via v1 API: {text[:50]}...")
                return tweet
            
            if not self.client_v2 and not self.api_v1:
                logger.error("No Twitter API client available")
                return None
                
        except Exception as e:
            logger.error(f"Error posting tweet: {str(e)}")
            return None
    
    def post_tweet_with_media(self, text, media_path):
        """Post a tweet with media attachment."""
        try:
            # Try using v1 API first as it handles media better
            if self.api_v1:
                try:
                    # Upload media
                    media = self.api_v1.media_upload(media_path)
                    
                    # Post tweet with media
                    tweet = self.api_v1.update_status(
                        status=text,
                        media_ids=[media.media_id]
                    )
                    
                    logger.info(f"Posted tweet with media via v1 API: {text[:50]}...")
                    return tweet
                except Exception as e:
                    logger.error(f"Error posting tweet with media via v1 API: {str(e)}")
            
            # Fall back to v2 API (though media handling is more complex)
            if self.client_v2:
                try:
                    # For v2, we first need to upload media via v1.1 endpoint
                    if not self.api_v1:
                        # Create a temporary v1 client just for media upload
                        auth = tweepy.OAuth1UserHandler(
                            self.consumer_key, 
                            self.consumer_secret,
                            self.access_token, 
                            self.access_token_secret
                        )
                        temp_api = tweepy.API(auth)
                        media = temp_api.media_upload(media_path)
                    else:
                        media = self.api_v1.media_upload(media_path)
                    
                    # Then post the tweet with media_id via v2
                    response = self.client_v2.create_tweet(
                        text=text,
                        media_ids=[media.media_id_string]
                    )
                    
                    if response and hasattr(response, 'data'):
                        tweet_id = response.data.get('id')
                        logger.info(f"Posted tweet with media via v2 API (ID: {tweet_id}): {text[:50]}...")
                        return response
                    else:
                        logger.error(f"Unexpected response from v2 API: {response}")
                        
                except Exception as e:
                    logger.error(f"Error posting tweet with media via v2 API: {str(e)}")
            
            if not self.client_v2 and not self.api_v1:
                logger.error("No Twitter API client available")
                return None
                
        except Exception as e:
            logger.error(f"Error posting tweet with media: {str(e)}")
            return None
    
    def reply_to_tweet(self, tweet_id, text):
        """Reply to a specific tweet."""
        try:
            # Try v2 API first
            if self.client_v2:
                response = self.client_v2.create_tweet(
                    text=text,
                    in_reply_to_tweet_id=tweet_id
                )
                if response and hasattr(response, 'data'):
                    reply_id = response.data.get('id')
                    logger.info(f"Replied to tweet {tweet_id} via v2 API (ID: {reply_id}): {text[:50]}...")
                    return response
                else:
                    logger.error(f"Unexpected response from v2 API: {response}")
            
            # Fall back to v1 if v2 fails or isn't available
            if self.api_v1 and (not self.client_v2 or not response):
                tweet = self.api_v1.update_status(
                    status=text,
                    in_reply_to_status_id=tweet_id,
                    auto_populate_reply_metadata=True
                )
                logger.info(f"Replied to tweet {tweet_id} via v1 API: {text[:50]}...")
                return tweet
            
            if not self.client_v2 and not self.api_v1:
                logger.error("No Twitter API client available")
                return None
                
        except Exception as e:
            logger.error(f"Error replying to tweet: {str(e)}")
            return None
    
    def reply_with_media(self, tweet_id, text, media_path):
        """Reply to a tweet with media attachment."""
        try:
            # Try using v1 API first as it handles media better
            if self.api_v1:
                try:
                    # Upload media
                    media = self.api_v1.media_upload(media_path)
                    
                    # Post reply with media
                    tweet = self.api_v1.update_status(
                        status=text,
                        in_reply_to_status_id=tweet_id,
                        media_ids=[media.media_id],
                        auto_populate_reply_metadata=True
                    )
                    
                    logger.info(f"Replied to tweet {tweet_id} with media via v1 API: {text[:50]}...")
                    return tweet
                except Exception as e:
                    logger.error(f"Error replying with media via v1 API: {str(e)}")
            
            # Fall back to v2 API
            if self.client_v2:
                try:
                    # For v2, we first need to upload media via v1.1 endpoint
                    if not self.api_v1:
                        # Create a temporary v1 client just for media upload
                        auth = tweepy.OAuth1UserHandler(
                            self.consumer_key, 
                            self.consumer_secret,
                            self.access_token, 
                            self.access_token_secret
                        )
                        temp_api = tweepy.API(auth)
                        media = temp_api.media_upload(media_path)
                    else:
                        media = self.api_v1.media_upload(media_path)
                    
                    # Then post the reply with media_id via v2
                    response = self.client_v2.create_tweet(
                        text=text,
                        media_ids=[media.media_id_string],
                        in_reply_to_tweet_id=tweet_id
                    )
                    
                    if response and hasattr(response, 'data'):
                        reply_id = response.data.get('id')
                        logger.info(f"Replied to tweet {tweet_id} with media via v2 API (ID: {reply_id}): {text[:50]}...")
                        return response
                    else:
                        logger.error(f"Unexpected response from v2 API: {response}")
                        
                except Exception as e:
                    logger.error(f"Error replying with media via v2 API: {str(e)}")
            
            if not self.client_v2 and not self.api_v1:
                logger.error("No Twitter API client available")
                return None
                
        except Exception as e:
            logger.error(f"Error replying to tweet with media: {str(e)}")
            return None
    
    def get_mentions(self, since_id=None):
        """Get mentions of the bot's username."""
        try:
            if self.client_v2:
                # Get user ID first
                me = self.client_v2.get_me()
                user_id = me.data.id
                
                # Get mentions
                mentions_response = self.client_v2.get_users_mentions(
                    id=user_id,
                    since_id=since_id,
                    max_results=100,
                    expansions=["author_id", "referenced_tweets.id"],
                    tweet_fields=["created_at", "text"]
                )
                
                if mentions_response.data:
                    logger.info(f"Fetched {len(mentions_response.data)} mentions via v2 API")
                    return mentions_response.data
                else:
                    logger.info("No mentions found via v2 API")
                    return []
            
            # Fall back to v1 API
            if self.api_v1:
                mentions = self.api_v1.mentions_timeline(since_id=since_id, tweet_mode='extended')
                logger.info(f"Fetched {len(mentions)} mentions via v1 API")
                return mentions
                    
            if not self.client_v2 and not self.api_v1:
                logger.error("No Twitter API client available")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching mentions: {str(e)}")
            return []
    
    def send_direct_message(self, user_id_or_screen_name, text):
        """Send a direct message to a user."""
        try:
            # Try v1 API as it's simpler for DMs
            if self.api_v1:
                try:
                    # If screen_name is provided instead of ID
                    if isinstance(user_id_or_screen_name, str) and not user_id_or_screen_name.isdigit():
                        # Get user ID from screen name
                        user = self.api_v1.get_user(screen_name=user_id_or_screen_name)
                        user_id = user.id
                    else:
                        user_id = user_id_or_screen_name
                        
                    self.api_v1.send_direct_message(recipient_id=user_id, text=text)
                    logger.info(f"Sent DM to user {user_id_or_screen_name}: {text[:50]}...")
                    return True
                except Exception as e:
                    logger.error(f"Error sending DM via v1 API: {str(e)}")
            
            # Fall back to v2 API if needed (more complex for DMs)
            logger.warning("Sending DMs via v2 API is not implemented yet")
            return False
            
        except Exception as e:
            logger.error(f"Error sending direct message: {str(e)}")
            return False