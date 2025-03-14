import tweepy
import logging
import time
import os
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class TwitterClient:
    def __init__(self, 
                 consumer_key: str, 
                 consumer_secret: str, 
                 access_token: str, 
                 access_token_secret: str,
                 bearer_token: Optional[str] = None):
        
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.bearer_token = bearer_token
        
        # Initialize API v1.1 client (needed for some features)
        auth = tweepy.OAuth1UserHandler(
            consumer_key, consumer_secret, access_token, access_token_secret
        )
        self.api = tweepy.API(auth)
        
        # Initialize API v2 client
        self.client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret
        )
        
        # Test Twitter connection
        try:
            self.client.get_me()
            logger.info("Twitter API v2 authentication successful")
        except Exception as e:
            logger.error(f"Twitter API v2 authentication failed: {str(e)}")
            
        try:
            self.api.verify_credentials()
            logger.info("Twitter API v1.1 authentication successful")
        except Exception as e:
            logger.error(f"Twitter API v1.1 authentication failed: {str(e)}")

    def post_tweet(self, text: str) -> Optional[str]:
        """Post a tweet using Twitter API v2 with error validation."""
        # Basic validation to prevent posting error messages
        if text.startswith("Error:") or "Error generating" in text:
            logger.error(f"Prevented posting an error message as a tweet: {text[:50]}...")
            return None
            
        # Ensure the content is not empty or just whitespace
        if not text or text.isspace():
            logger.error("Prevented posting empty or whitespace-only tweet")
            return None
            
        try:
            response = self.client.create_tweet(text=text)
            tweet_id = response.data["id"]
            logger.info(f"Posted tweet via v2 API (ID: {tweet_id}): {text[:50]}...")
            return tweet_id
        except Exception as e:
            logger.error(f"Error posting tweet: {str(e)}")
            return None

    def post_tweet_with_media(self, text: str, media_path: str) -> Optional[str]:
        """Post a tweet with media using Twitter API v1.1."""
        # Basic validation to prevent posting error messages
        if text.startswith("Error:") or "Error generating" in text:
            logger.error(f"Prevented posting an error message as a tweet: {text[:50]}...")
            return None
            
        # Ensure the content is not empty or just whitespace
        if not text or text.isspace():
            logger.error("Prevented posting empty or whitespace-only tweet")
            return None
            
        # Validate media file exists
        if not os.path.exists(media_path):
            logger.error(f"Media file not found: {media_path}")
            return None
            
        try:
            # Upload media using v1.1 API
            media = self.api.media_upload(media_path)
            media_id = media.media_id_string
            
            # Post tweet with media using v2 API
            response = self.client.create_tweet(text=text, media_ids=[media_id])
            tweet_id = response.data["id"]
            logger.info(f"Posted tweet with media via v2 API (ID: {tweet_id}): {text[:50]}...")
            return tweet_id
        except Exception as e:
            logger.error(f"Error posting tweet with media: {str(e)}")
            return None

    def reply_to_tweet(self, tweet_id: str, text: str) -> Optional[str]:
        """Reply to a tweet using Twitter API v2."""
        # Basic validation to prevent posting error messages
        if text.startswith("Error:") or "Error generating" in text:
            logger.error(f"Prevented posting an error message as a reply: {text[:50]}...")
            return None
            
        # Ensure the content is not empty or just whitespace
        if not text or text.isspace():
            logger.error("Prevented posting empty or whitespace-only reply")
            return None
            
        try:
            response = self.client.create_tweet(text=text, in_reply_to_tweet_id=tweet_id)
            reply_id = response.data["id"]
            logger.info(f"Replied to tweet {tweet_id} via v2 API (ID: {reply_id}): {text[:50]}...")
            return reply_id
        except Exception as e:
            logger.error(f"Error replying to tweet: {str(e)}")
            return None

    def reply_to_tweet_with_media(self, tweet_id: str, text: str, media_path: str) -> Optional[str]:
        """Reply to a tweet with media using Twitter API v1.1 and v2."""
        # Basic validation to prevent posting error messages
        if text.startswith("Error:") or "Error generating" in text:
            logger.error(f"Prevented posting an error message as a reply: {text[:50]}...")
            return None
            
        # Ensure the content is not empty or just whitespace
        if not text or text.isspace():
            logger.error("Prevented posting empty or whitespace-only reply")
            return None
            
        # Validate media file exists
        if not os.path.exists(media_path):
            logger.error(f"Media file not found: {media_path}")
            return None
            
        try:
            # Upload media using v1.1 API
            media = self.api.media_upload(media_path)
            media_id = media.media_id_string
            
            # Reply to tweet with media using v2 API
            response = self.client.create_tweet(
                text=text, 
                media_ids=[media_id],
                in_reply_to_tweet_id=tweet_id
            )
            reply_id = response.data["id"]
            logger.info(f"Replied to tweet {tweet_id} with media via v2 API (ID: {reply_id}): {text[:50]}...")
            return reply_id
        except Exception as e:
            logger.error(f"Error replying to tweet with media: {str(e)}")
            return None

    def get_user_tweets(self, username: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent tweets from a specific user."""
        try:
            # Get user ID first
            user = self.client.get_user(username=username)
            if not user or not user.data:
                logger.error(f"User {username} not found")
                return []
            
            user_id = user.data.id
            
            # Get user tweets
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=count,
                tweet_fields=["created_at", "public_metrics", "text"],
                expansions=["author_id"],
                user_fields=["name", "username", "profile_image_url"]
            )
            
            if not tweets or not tweets.data:
                logger.info(f"No tweets found for {username}")
                return []
            
            processed_tweets = []
            for tweet in tweets.data:
                processed_tweets.append({
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "likes": tweet.public_metrics["like_count"],
                    "retweets": tweet.public_metrics["retweet_count"],
                    "username": username
                })
            
            logger.info(f"Retrieved {len(processed_tweets)} tweets from {username}")
            return processed_tweets
        except Exception as e:
            logger.error(f"Error fetching tweets from {username}: {str(e)}")
            return []

    def search_tweets(self, query: str, count: int = 20) -> List[Dict[str, Any]]:
        """Search for tweets matching a specific query."""
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=count,
                tweet_fields=["created_at", "public_metrics", "text"],
                expansions=["author_id"],
                user_fields=["name", "username", "profile_image_url"]
            )
            
            if not tweets or not tweets.data:
                logger.info(f"No tweets found for query: {query}")
                return []
            
            # Create user lookup dictionary
            users = {user.id: user for user in tweets.includes["users"]} if "users" in tweets.includes else {}
            
            processed_tweets = []
            for tweet in tweets.data:
                user = users.get(tweet.author_id)
                username = user.username if user else "unknown"
                
                processed_tweets.append({
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "likes": tweet.public_metrics["like_count"],
                    "retweets": tweet.public_metrics["retweet_count"],
                    "username": username
                })
            
            logger.info(f"Retrieved {len(processed_tweets)} tweets for query: {query}")
            return processed_tweets
        except Exception as e:
            logger.error(f"Error searching tweets for '{query}': {str(e)}")
            return []

    def get_mentions(self, since_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get mentions of the authenticated user."""
        try:
            kwargs = {
                "expansions": ["author_id", "referenced_tweets.id"],
                "user_fields": ["name", "username"],
                "max_results": 100
            }
            
            if since_id:
                kwargs["since_id"] = since_id
            
            # Get current user ID
            me = self.client.get_me()
            if not me or not me.data:
                logger.error("Failed to get current user information")
                return []
                
            user_id = me.data.id
            
            # Get mentions
            mentions = self.client.get_users_mentions(id=user_id, **kwargs)
            
            if not mentions or not mentions.data:
                logger.info("No new mentions found")
                return []
                
            processed_mentions = []
            
            # Create user lookup dictionary
            users = {user.id: user for user in mentions.includes["users"]} if "users" in mentions.includes and "users" in mentions.includes else {}
            
            for mention in mentions.data:
                user = users.get(mention.author_id)
                username = user.username if user else "unknown"
                
                processed_mentions.append({
                    "id": mention.id,
                    "text": mention.text,
                    "created_at": mention.created_at,
                    "author_id": mention.author_id,
                    "username": username
                })
            
            logger.info(f"Retrieved {len(processed_mentions)} mentions")
            return processed_mentions
        except Exception as e:
            logger.error(f"Error fetching mentions: {str(e)}")
            return []

    def create_thread(self, tweets_content: List[str]) -> Tuple[bool, List[str]]:
        """Create a thread of tweets with improved validation."""
        if not tweets_content:
            logger.error("No content provided for thread")
            return False, []
        
        # Validate each tweet in the thread
        valid_tweets = []
        for content in tweets_content:
            # Skip empty tweets or error messages
            if not content or content.isspace() or content.startswith("Error:") or "Error generating" in content:
                logger.warning(f"Skipping invalid tweet content in thread: {content[:50]}...")
                continue
            valid_tweets.append(content)
            
        if not valid_tweets:
            logger.error("No valid tweets found in thread content")
            return False, []
                
        tweet_ids = []
        previous_tweet_id = None
        
        for content in valid_tweets:
            try:
                if previous_tweet_id is None:
                    # First tweet in thread
                    response = self.client.create_tweet(text=content)
                    tweet_id = response.data["id"]
                else:
                    # Reply to create thread
                    response = self.client.create_tweet(
                        text=content,
                        in_reply_to_tweet_id=previous_tweet_id
                    )
                    tweet_id = response.data["id"]
                
                tweet_ids.append(tweet_id)
                previous_tweet_id = tweet_id
                logger.info(f"Posted tweet in thread (ID: {tweet_id}): {content[:50]}...")
                
                # Sleep briefly to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error creating thread: {str(e)}")
                return False, tweet_ids
                
        return True, tweet_ids