#!/usr/bin/env python3

import os
import sys
import time
import json
import logging
import requests
import schedule
import tweepy
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
import fcntl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Twitter API credentials
TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# News API key
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# CoinGecko API base URL
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Rate limit file path - shared between bots
RATE_LIMIT_FILE = "/tmp/twitter_rate_limits.json"

# OpenAI Helper class - using older version for compatibility
class OpenAIHelper:
    def __init__(self):
        # Import here to avoid issues if not using this class
        import openai
        openai.api_key = OPENAI_API_KEY
        self.openai = openai
    
    def generate_content(self, prompt):
        try:
            response = self.openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating AI content: {str(e)}")
            return None

# Global rate limit manager
class GlobalRateLimitManager:
    """Manage Twitter API rate limits across multiple bots"""
    
    def __init__(self, file_path=RATE_LIMIT_FILE):
        self.file_path = file_path
        self.lock_timeout = 5  # seconds
        
        # Initialize file if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump({
                    "last_reset": time.time(),
                    "window_length": 15 * 60,  # 15 minutes in seconds
                    "requests": {
                        "mentions": 0,
                        "timeline": 0,
                        "tweets": 0,
                        "other": 0
                    },
                    "limits": {
                        "mentions": 60,  # Reduced from 75
                        "timeline": 60,  # Reduced from 75
                        "tweets": 240,   # Reduced from 300
                        "other": 180
                    }
                }, f)
    
    def _get_lock(self, file_obj):
        """Get an exclusive lock on the file"""
        start_time = time.time()
        while True:
            try:
                fcntl.flock(file_obj, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except IOError:
                # Check if we've timed out
                if time.time() - start_time > self.lock_timeout:
                    logger.error("Timed out waiting for lock on rate limit file")
                    return False
                time.sleep(0.1)
    
    def _release_lock(self, file_obj):
        """Release the lock"""
        fcntl.flock(file_obj, fcntl.LOCK_UN)
    
    def can_make_request(self, request_type):
        """Check if a request can be made without exceeding rate limits"""
        try:
            with open(self.file_path, 'r+') as f:
                if not self._get_lock(f):
                    logger.warning("Failed to get lock, assuming rate limit exceeded")
                    return False
                
                data = json.load(f)
                
                # Check if we need to reset counters
                current_time = time.time()
                if current_time - data["last_reset"] > data["window_length"]:
                    data["last_reset"] = current_time
                    for key in data["requests"]:
                        data["requests"][key] = 0
                
                # Check if we can make the request
                if request_type not in data["requests"]:
                    request_type = "other"
                
                can_request = data["requests"][request_type] < data["limits"][request_type]
                
                # Rewind file and update
                f.seek(0)
                json.dump(data, f)
                f.truncate()
                
                self._release_lock(f)
                return can_request
                
        except Exception as e:
            logger.error(f"Error checking rate limits: {str(e)}")
            return False
    
    def register_request(self, request_type):
        """Register that a request was made"""
        try:
            with open(self.file_path, 'r+') as f:
                if not self._get_lock(f):
                    logger.warning("Failed to get lock, couldn't register request")
                    return
                
                data = json.load(f)
                
                # Check if we need to reset counters
                current_time = time.time()
                if current_time - data["last_reset"] > data["window_length"]:
                    data["last_reset"] = current_time
                    for key in data["requests"]:
                        data["requests"][key] = 0
                
                # Register the request
                if request_type not in data["requests"]:
                    request_type = "other"
                
                data["requests"][request_type] += 1
                
                # Rewind file and update
                f.seek(0)
                json.dump(data, f)
                f.truncate()
                
                self._release_lock(f)
                
        except Exception as e:
            logger.error(f"Error registering request: {str(e)}")

# Twitter client class
class TwitterClient:
    """Twitter API client with rate limit handling"""
    
    def __init__(self, rate_limit_manager=None):
        self.rate_limit_manager = rate_limit_manager
        self.processed_mentions = set()
        self.setup_clients()
        self.bot_username = "hedgehogai2025"  # Your actual bot username
        self.can_post_tweets = True  # Always set to True based on our test
        self.can_read_mentions = False  # Set to False since we know from logs that this doesn't work
        
    def setup_clients(self):
        """Set up Twitter API v1 and v2 clients"""
        try:
            # Twitter API v1 for tweeting
            auth = tweepy.OAuth1UserHandler(
                TWITTER_CONSUMER_KEY,
                TWITTER_CONSUMER_SECRET,
                TWITTER_ACCESS_TOKEN,
                TWITTER_ACCESS_TOKEN_SECRET
            )
            self.api_v1 = tweepy.API(auth)
            
            # Twitter API v2 for additional functionality
            self.api_v2 = tweepy.Client(
                consumer_key=TWITTER_CONSUMER_KEY,
                consumer_secret=TWITTER_CONSUMER_SECRET,
                access_token=TWITTER_ACCESS_TOKEN,
                access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
            )
            
            # Force posting permission to true since we've confirmed it works
            self.can_post_tweets = True
            
            logger.info("Twitter clients set up successfully")
        except Exception as e:
            logger.error(f"Error setting up Twitter clients: {str(e)}")
            raise
    
    def get_mentions(self, max_results=10):
        """Get recent mentions with exponential backoff for rate limits"""
        # If we already know we don't have permission, don't try
        if not self.can_read_mentions:
            logger.info("Skipping mentions check due to known permission issues")
            return []
            
        max_retries = 5
        retry = 0
        
        while retry < max_retries:
            try:
                if self.rate_limit_manager and not self.rate_limit_manager.can_make_request("mentions"):
                    logger.warning("Rate limit would be exceeded. Skipping mentions check.")
                    return []
                
                # Get mentions using Tweepy API v1
                # Use a lower count to reduce API usage
                mentions = self.api_v1.mentions_timeline(count=max(5, max_results))
                
                if self.rate_limit_manager:
                    self.rate_limit_manager.register_request("mentions")
                
                if mentions:
                    return mentions
                return []
                
            except Exception as e:
                if "429" in str(e):
                    wait_time = (2 ** retry) * 60  # Increased exponential backoff
                    logger.warning(f"Rate limited. Waiting for {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    retry += 1
                elif "403" in str(e):
                    # Handle permission issues more gracefully
                    logger.error(f"Permission error fetching mentions: {str(e)}")
                    self.can_read_mentions = False  # Set flag to avoid future tries
                    return []
                else:
                    logger.error(f"Error fetching mentions: {str(e)}")
                    logger.error(str(e))
                    return []
        
        logger.error("Max retries reached while fetching mentions")
        return []
    
    def post_tweet(self, text):
        """Post a tweet with rate limit handling using v2 API"""
        # If we already know we don't have permission, don't try
        if not self.can_post_tweets:
            logger.info("Skipping tweet posting due to known permission issues")
            return None
            
        max_retries = 3
        retry = 0
        
        while retry < max_retries:
            try:
                if self.rate_limit_manager and not self.rate_limit_manager.can_make_request("tweets"):
                    wait_time = 300 + (retry * 300)  # Increased wait time on subsequent retries
                    logger.warning(f"Rate limit would be exceeded. Delaying tweet for {wait_time} seconds.")
                    time.sleep(wait_time)
                    retry += 1
                    continue
                
                logger.info(f"Attempting to post tweet via v2 API: {text[:50]}...")
                
                # Use v2 API for tweet posting
                response = self.api_v2.create_tweet(text=text)
                
                # Enhanced logging of the response
                logger.info(f"Twitter API v2 response data: {response.data}")
                
                if self.rate_limit_manager:
                    self.rate_limit_manager.register_request("tweets")
                
                logger.info(f"Tweet posted successfully via v2 API: {text[:50]}... with ID: {response.data['id']}")
                
                # Create a response object compatible with the rest of the code
                class TweetResponse:
                    def __init__(self, id):
                        self.id = id
                
                return TweetResponse(response.data['id'])
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error posting tweet: {error_msg}")
                
                if "429" in error_msg:
                    wait_time = (2 ** retry) * 120  # Increased exponential backoff
                    logger.warning(f"Rate limited. Waiting for {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    retry += 1
                elif "403" in error_msg:
                    logger.error(f"Permission error posting tweet: {error_msg}")
                    logger.error(f"Tweet content that failed: {text}")
                    # We keep can_post_tweets as True since we know it works
                    return None
                else:
                    logger.error(f"Unexpected error posting tweet: {error_msg}")
                    logger.error(f"Tweet content that failed: {text}")
                    return None
        
        logger.error("Max retries reached while posting tweet")
        return None
    
    def post_reply(self, text, tweet_id, username):
        """Post a reply to another user's tweet with rate limit handling"""
        # If we already know we don't have permission, don't try
        if not self.can_post_tweets:
            logger.info("Skipping reply posting due to known permission issues")
            return None
            
        max_retries = 3
        retry = 0
        
        # Format the reply to include the username
        reply_text = f"@{username} {text}"
        
        while retry < max_retries:
            try:
                if self.rate_limit_manager and not self.rate_limit_manager.can_make_request("tweets"):
                    wait_time = 300 + (retry * 300)
                    logger.warning(f"Rate limit would be exceeded. Delaying reply for {wait_time} seconds.")
                    time.sleep(wait_time)
                    retry += 1
                    continue
                
                logger.info(f"Attempting to post reply via v2 API: {reply_text[:50]}...")
                
                # Use v2 API for reply posting
                response = self.api_v2.create_tweet(
                    text=reply_text,
                    in_reply_to_tweet_id=tweet_id
                )
                
                logger.info(f"Twitter API v2 response data for reply: {response.data}")
                
                if self.rate_limit_manager:
                    self.rate_limit_manager.register_request("tweets")
                
                logger.info(f"Reply posted successfully via v2 API: {reply_text[:50]}... with ID: {response.data['id']}")
                
                # Create a response object compatible with the rest of the code
                class TweetResponse:
                    def __init__(self, id):
                        self.id = id
                
                return TweetResponse(response.data['id'])
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error posting reply: {error_msg}")
                
                if "429" in error_msg:
                    wait_time = (2 ** retry) * 120
                    logger.warning(f"Rate limited. Waiting for {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    retry += 1
                elif "403" in error_msg:
                    logger.error(f"Permission error posting reply: {error_msg}")
                    logger.error(f"Reply content that failed: {reply_text}")
                    # We keep can_post_tweets as True since we know it works
                    return None
                else:
                    logger.error(f"Unexpected error posting reply: {error_msg}")
                    logger.error(f"Reply content that failed: {reply_text}")
                    return None
        
        logger.error("Max retries reached while posting reply")
        return None
    
    def post_thread(self, tweets_list):
        """Post a thread of tweets with rate limit handling"""
        # If we already know we don't have permission, don't try
        if not self.can_post_tweets:
            logger.info("Skipping thread posting due to known permission issues")
            return
            
        if not tweets_list:
            logger.warning("Empty thread. Nothing to post.")
            return
        
        logger.info(f"Starting to post thread with {len(tweets_list)} tweets")
        
        previous_tweet = None
        
        for i, tweet_text in enumerate(tweets_list):
            if previous_tweet is None:
                # First tweet in thread
                logger.info(f"Posting first tweet in thread: {tweet_text[:50]}...")
                response = self.post_tweet(tweet_text)
                if response:
                    previous_tweet = response
                    logger.info(f"First tweet in thread posted successfully with ID: {response.id}")
                else:
                    logger.error("Failed to post first tweet in thread. Aborting thread.")
                    return
            else:
                # Thread continuation - use reply without adding username in text
                logger.info(f"Posting tweet #{i+1} in thread as reply to tweet {previous_tweet.id}")
                
                # Use v2 API for thread continuation
                try:
                    # Do not add username to the text for thread continuation
                    response = self.api_v2.create_tweet(
                        text=tweet_text,  # Do not add @username prefix
                        in_reply_to_tweet_id=previous_tweet.id
                    )
                    
                    # Log success
                    logger.info(f"Thread tweet #{i+1} posted successfully with ID: {response.data['id']}")
                    
                    # Create a response object
                    class TweetResponse:
                        def __init__(self, id):
                            self.id = id
                    
                    previous_tweet = TweetResponse(response.data['id'])
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error posting thread continuation: {error_msg}")
                    logger.error(f"Tweet content that failed: {tweet_text}")
                    return
            
            # Sleep between tweets to avoid rate limits
            logger.info(f"Sleeping for 10 seconds between tweets in thread")
            time.sleep(10)
        
        logger.info(f"Thread with {len(tweets_list)} tweets posted successfully")

# Crypto data fetching class
class CryptoDataClient:
    """Client for fetching cryptocurrency data"""
    
    def __init__(self):
        self.base_url = COINGECKO_API_URL
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 5  # seconds between requests to prevent rate limiting
    
    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def get_market_data(self, vs_currency="usd", count=10):
        """Get top cryptocurrency market data"""
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/coins/markets"
            params = {
                "vs_currency": vs_currency,
                "order": "market_cap_desc",
                "per_page": count,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h,7d,30d"
            }
            
            logger.info(f"Fetching market data from CoinGecko API")
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched market data for {len(data)} coins")
                return data
            elif response.status_code == 429:
                logger.warning("Rate limited by CoinGecko API. Will retry later.")
                time.sleep(60)  # Sleep for a minute before the next attempt
                return None
            else:
                logger.error(f"Error fetching market data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None
    
    def get_global_data(self):
        """Get global crypto market data"""
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/global"
            logger.info(f"Fetching global market data from CoinGecko API")
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched global market data")
                return data
            elif response.status_code == 429:
                logger.warning("Rate limited by CoinGecko API. Will retry later.")
                time.sleep(60)  # Sleep for a minute before the next attempt
                return None
            else:
                logger.error(f"Error fetching global data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching global data: {str(e)}")
            return None
    
    def get_coin_data(self, coin_id):
        """Get detailed data for a specific coin"""
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                "localization": False,
                "tickers": False,
                "market_data": True,
                "community_data": False,
                "developer_data": False
            }
            
            logger.info(f"Fetching detailed data for coin: {coin_id}")
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched detailed data for {coin_id}")
                return data
            elif response.status_code == 429:
                logger.warning("Rate limited by CoinGecko API. Will retry later.")
                time.sleep(60)  # Sleep for a minute before the next attempt
                return None
            else:
                logger.error(f"Error fetching coin data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching coin data: {str(e)}")
            return None

# News data client
class NewsDataClient:
    """Client for fetching news data"""
    
    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2"
        self.session = requests.Session()
        self.last_request_time = 0
        self.min_request_interval = 2  # seconds between requests
    
    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def get_crypto_news(self, max_results=5):
        """Get latest cryptocurrency news"""
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/everything"
            params = {
                "q": "cryptocurrency OR bitcoin OR blockchain",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_results,
                "apiKey": self.api_key
            }
            
            logger.info(f"Fetching crypto news from News API")
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok" and data.get("totalResults", 0) > 0:
                    articles = data.get("articles", [])
                    logger.info(f"Successfully fetched {len(articles)} crypto news articles")
                    return articles
                logger.warning("No crypto news articles found")
                return []
            else:
                logger.error(f"Error fetching crypto news: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching crypto news: {str(e)}")
            return []
    
    def get_ai_news(self, max_results=5):
        """Get latest AI news"""
        try:
            self._wait_for_rate_limit()
            
            url = f"{self.base_url}/everything"
            params = {
                "q": "artificial intelligence OR machine learning OR AI technology",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_results,
                "apiKey": self.api_key
            }
            
            logger.info(f"Fetching AI news from News API")
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok" and data.get("totalResults", 0) > 0:
                    articles = data.get("articles", [])
                    logger.info(f"Successfully fetched {len(articles)} AI news articles")
                    return articles
                logger.warning("No AI news articles found")
                return []
            else:
                logger.error(f"Error fetching AI news: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching AI news: {str(e)}")
            return []

# Content generator class
class AIXBTStyleContentGenerator:
    """Generate content similar to @aixbt_agent"""
    
    def __init__(self, ai_client, crypto_client, news_client):
        self.ai_client = ai_client
        self.crypto_client = crypto_client
        self.news_client = news_client
        self.topics = [
            "cryptocurrency news", 
            "blockchain technology", 
            "AI developments", 
            "machine learning", 
            "crypto market analysis",
            "web3 innovations",
            "defi updates"
        ]
        # Add randomness to avoid content duplication between bots
        self.bot_id = f"advanced-bot-{random.randint(1000, 9999)}"
    
    def generate_market_update(self):
        """Generate market update content"""
        logger.info("Starting market update content generation")
        # Get market data
        market_data = self.crypto_client.get_market_data(count=5)
        global_data = self.crypto_client.get_global_data()
        
        # Check if data was successfully retrieved
        if not market_data or not global_data:
            logger.error("Failed to fetch data for market update")
            return None
        
        logger.info(f"Successfully retrieved market data with {len(market_data)} coins")
        
        try:
            # Prepare market summary
            market_summary = {
                "total_market_cap": global_data["data"]["total_market_cap"]["usd"],
                "market_cap_change": global_data["data"]["market_cap_change_percentage_24h_usd"],
                "btc_dominance": global_data["data"]["market_cap_percentage"]["btc"]
            }
            
            logger.info(f"Market summary prepared: Market cap: ${market_summary['total_market_cap']:.2f}, Change: {market_summary['market_cap_change']:.2f}%")
            
            # Prepare top coins data
            top_coins = []
            for coin in market_data:
                top_coins.append({
                    "name": coin["name"],
                    "symbol": coin["symbol"].upper(),
                    "price": coin["current_price"],
                    "change_24h": coin["price_change_percentage_24h"],
                    "change_7d": coin["price_change_percentage_7d_in_currency"] if "price_change_percentage_7d_in_currency" in coin else None
                })
            
            # Add random focus for variety
            focus_coins = random.sample(top_coins, min(3, len(top_coins)))
            
            logger.info(f"Selected focus coins: {', '.join([coin['symbol'] for coin in focus_coins])}")
            
            # Create prompt for AI
            prompt = f"""
            Generate a Twitter thread (3-4 tweets) with a market update for cryptocurrency. 
            Include the following data:
            
            Global Market:
            - Total Market Cap: ${market_summary['total_market_cap']:.2f}
            - 24h Change: {market_summary['market_cap_change']:.2f}%
            - BTC Dominance: {market_summary['btc_dominance']:.2f}%
            
            Top Performing Coins:
            {', '.join([f"{coin['symbol']}: ${coin['price']:.2f} ({coin['change_24h']:.2f}%)" for coin in focus_coins])}
            
            Format as engaging tweets with relevant hashtags like #crypto #bitcoin. 
            Similar to @aixbt_agent style - professional yet conversational.
            Each tweet should be under 280 characters.
            
            Bot identifier: {self.bot_id}
            """
            
            logger.info("Sending prompt to OpenAI for content generation")
            
            # Generate content
            content = self.ai_client.generate_content(prompt)
            
            if not content:
                logger.error("OpenAI returned empty content")
                return None
            
            logger.info(f"Successfully generated content from OpenAI: {len(content)} characters")
            logger.info(f"First 100 chars of content: {content[:100]}")
            
            # Split into tweets
            tweets = self._format_as_thread(content)
            logger.info(f"Formatted content into {len(tweets)} tweets")
            
            for i, tweet in enumerate(tweets):
                logger.info(f"Tweet {i+1}: {tweet[:50]}... ({len(tweet)} chars)")
            
            return tweets
            
        except Exception as e:
            logger.error(f"Error generating market update: {str(e)}")
            logger.exception("Full exception details:")
            return None
    
    def generate_ai_news(self):
        """Generate AI industry news update"""
        logger.info("Starting AI news content generation")
        # Get AI news
        news_articles = self.news_client.get_ai_news(max_results=3)
        
        if not news_articles:
            logger.warning("No AI news articles found")
            
        # Prepare news data
        news_data = []
        for article in news_articles:
            news_data.append({
                "title": article.get("title", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", "")
            })
        
        logger.info(f"Prepared {len(news_data)} AI news items for content generation")
        
        # Create prompt for AI
        if news_data:
            prompt = f"""
            Generate a Twitter thread (3 tweets) about the latest developments in AI industry.
            Include these recent news items:
            
            {", ".join([f"'{item['title']}' by {item['source']}" for item in news_data])}
            
            Make it informative yet engaging with relevant hashtags like #AI #MachineLearning.
            Similar to @aixbt_agent style - professional yet conversational.
            Each tweet should be under 280 characters.
            Include one major development and its potential impact.
            
            Bot identifier: {self.bot_id}
            """
        else:
            prompt = f"""
            Generate a Twitter thread (3 tweets) about the latest developments in AI industry.
            Focus on recent advancements, new research papers, and applications.
            
            Make it informative yet engaging with relevant hashtags like #AI #MachineLearning.
            Similar to @aixbt_agent style - professional yet conversational.
            Each tweet should be under 280 characters.
            Include one major development and its potential impact.
            
            Bot identifier: {self.bot_id}
            """
        
        logger.info("Sending prompt to OpenAI for AI news content generation")
        
        # Generate content
        content = self.ai_client.generate_content(prompt)
        
        if not content:
            logger.error("OpenAI returned empty content for AI news")
            return None
        
        logger.info(f"Successfully generated AI news content: {len(content)} characters")
        logger.info(f"First 100 chars of AI news content: {content[:100]}")
        
        # Split into tweets
        tweets = self._format_as_thread(content)
        logger.info(f"Formatted AI news content into {len(tweets)} tweets")
        
        for i, tweet in enumerate(tweets):
            logger.info(f"AI News Tweet {i+1}: {tweet[:50]}... ({len(tweet)} chars)")
        
        return tweets
    
    def generate_technical_analysis(self):
        """Generate technical analysis for Bitcoin or Ethereum"""
        logger.info("Starting technical analysis content generation")
        # Randomly choose between Bitcoin and Ethereum
        coin = random.choice(["bitcoin", "ethereum"])
        logger.info(f"Selected {coin} for technical analysis")
        
        coin_data = self.crypto_client.get_coin_data(coin)
        
        if not coin_data:
            logger.error(f"Failed to fetch data for {coin}")
            return None
        
        logger.info(f"Successfully retrieved detailed data for {coin}")
        
        try:
            # Extract relevant data
            name = coin_data["name"]
            symbol = coin_data["symbol"].upper()
            current_price = coin_data["market_data"]["current_price"]["usd"]
            price_change_24h = coin_data["market_data"]["price_change_percentage_24h"]
            price_change_7d = coin_data["market_data"]["price_change_percentage_7d"]
            price_change_30d = coin_data["market_data"]["price_change_percentage_30d"]
            ath = coin_data["market_data"]["ath"]["usd"]
            ath_change_percentage = coin_data["market_data"]["ath_change_percentage"]["usd"]
            
            logger.info(f"Prepared data for {name} technical analysis")
            
            prompt = f"""
            Generate a technical analysis Twitter thread (3-4 tweets) for {name} ({symbol}).
            
            Price data:
            - Current Price: ${current_price:.2f}
            - 24h Change: {price_change_24h:.2f}%
            - 7d Change: {price_change_7d:.2f}%
            - 30d Change: {price_change_30d:.2f}%
            - ATH: ${ath:.2f} ({ath_change_percentage:.2f}% from ATH)
            
            Include analysis of support/resistance levels, potential trend directions, and trading volume.
            Format as engaging tweets with relevant hashtags like #{symbol} #crypto #TechnicalAnalysis.
            Similar to @aixbt_agent style - professional yet analytical.
            Each tweet should be under 280 characters.
            
            Bot identifier: {self.bot_id}
            """
            
            logger.info("Sending prompt to OpenAI for technical analysis content generation")
            
            # Generate content
            content = self.ai_client.generate_content(prompt)
            
            if not content:
                logger.error("OpenAI returned empty content for technical analysis")
                return None
            
            logger.info(f"Successfully generated technical analysis content: {len(content)} characters")
            logger.info(f"First 100 chars of technical analysis: {content[:100]}")
            
            # Split into tweets
            tweets = self._format_as_thread(content)
            logger.info(f"Formatted technical analysis into {len(tweets)} tweets")
            
            for i, tweet in enumerate(tweets):
                logger.info(f"Technical Analysis Tweet {i+1}: {tweet[:50]}... ({len(tweet)} chars)")
            
            return tweets
            
        except Exception as e:
            logger.error(f"Error generating technical analysis: {str(e)}")
            logger.exception("Full exception details:")
            return None
    
    def generate_crypto_news(self):
        """Generate cryptocurrency news update"""
        logger.info("Starting crypto news content generation")
        # Get crypto news
        news_articles = self.news_client.get_crypto_news(max_results=3)
        
        if not news_articles:
            logger.warning("No crypto news articles found")
            
        # Prepare news data
        news_data = []
        for article in news_articles:
            news_data.append({
                "title": article.get("title", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", "")
            })
        
        logger.info(f"Prepared {len(news_data)} crypto news items for content generation")
        
        # Create prompt for AI
        if news_data:
            prompt = f"""
            Generate a Twitter thread (3 tweets) about the latest cryptocurrency news.
            Include these recent news items:
            
            {", ".join([f"'{item['title']}' by {item['source']}" for item in news_data])}
            
            Focus on regulatory updates, notable projects, and industry trends.
            Make it informative yet engaging with relevant hashtags like #crypto #blockchain.
            Similar to @aixbt_agent style - professional yet conversational.
            Each tweet should be under 280 characters.
            
            Bot identifier: {self.bot_id}
            """
        else:
            prompt = f"""
            Generate a Twitter thread (3 tweets) about the latest cryptocurrency news.
            Focus on regulatory updates, notable projects, and industry trends.
            
            Make it informative yet engaging with relevant hashtags like #crypto #blockchain.
            Similar to @aixbt_agent style - professional yet conversational.
            Each tweet should be under 280 characters.
            Include one major development and its potential impact.
            
            Bot identifier: {self.bot_id}
            """
        
        logger.info("Sending prompt to OpenAI for crypto news content generation")
        
        # Generate content
        content = self.ai_client.generate_content(prompt)
        
        if not content:
            logger.error("OpenAI returned empty content for crypto news")
            return None
        
        logger.info(f"Successfully generated crypto news content: {len(content)} characters")
        logger.info(f"First 100 chars of crypto news: {content[:100]}")
        
        # Split into tweets
        tweets = self._format_as_thread(content)
        logger.info(f"Formatted crypto news into {len(tweets)} tweets")
        
        for i, tweet in enumerate(tweets):
            logger.info(f"Crypto News Tweet {i+1}: {tweet[:50]}... ({len(tweet)} chars)")
        
        return tweets
    
    def generate_market_recap(self):
        """Generate end-of-day market recap"""
        logger.info("Starting market recap content generation")
        market_data = self.crypto_client.get_market_data(count=10)
        global_data = self.crypto_client.get_global_data()
        
        if not market_data or not global_data:
            logger.error("Failed to fetch data for market recap")
            return None
        
        logger.info(f"Successfully retrieved market data for {len(market_data)} coins for recap")
        
        try:
            # Prepare market summary
            market_summary = {
                "total_market_cap": global_data["data"]["total_market_cap"]["usd"],
                "market_cap_change": global_data["data"]["market_cap_change_percentage_24h_usd"],
                "btc_dominance": global_data["data"]["market_cap_percentage"]["btc"]
            }
            
            logger.info(f"Market recap summary prepared: Market cap: ${market_summary['total_market_cap']:.2f}, Change: {market_summary['market_cap_change']:.2f}%")
            
            # Get winners and losers
            winners = sorted(market_data, key=lambda x: x["price_change_percentage_24h"], reverse=True)[:3]
            losers = sorted(market_data, key=lambda x: x["price_change_percentage_24h"])[:3]
            
            logger.info(f"Top gainers: {', '.join([coin['symbol'].upper() for coin in winners])}")
            logger.info(f"Top losers: {', '.join([coin['symbol'].upper() for coin in losers])}")
            
            # Create prompt for AI
            prompt = f"""
            Generate a Twitter thread (3-4 tweets) with an end-of-day recap for the cryptocurrency market. 
            Include the following data:
            
            Global Market:
            - Total Market Cap: ${market_summary['total_market_cap']:.2f}
            - 24h Change: {market_summary['market_cap_change']:.2f}%
            - BTC Dominance: {market_summary['btc_dominance']:.2f}%
            
            Top Gainers:
            {', '.join([f"{coin['symbol'].upper()}: {coin['price_change_percentage_24h']:.2f}%" for coin in winners])}
            
            Top Losers:
            {', '.join([f"{coin['symbol'].upper()}: {coin['price_change_percentage_24h']:.2f}%" for coin in losers])}
            
            Format as engaging tweets with relevant hashtags like #crypto #DailyRecap. 
            Similar to @aixbt_agent style - professional yet conversational.
            Each tweet should be under 280 characters.
            
            Bot identifier: {self.bot_id}
            """
            
            logger.info("Sending prompt to OpenAI for market recap content generation")
            
            # Generate content
            content = self.ai_client.generate_content(prompt)
            
            if not content:
                logger.error("OpenAI returned empty content for market recap")
                return None
            
            logger.info(f"Successfully generated market recap content: {len(content)} characters")
            logger.info(f"First 100 chars of market recap: {content[:100]}")
            
            # Split into tweets
            tweets = self._format_as_thread(content)
            logger.info(f"Formatted market recap into {len(tweets)} tweets")
            
            for i, tweet in enumerate(tweets):
                logger.info(f"Market Recap Tweet {i+1}: {tweet[:50]}... ({len(tweet)} chars)")
            
            return tweets
            
        except Exception as e:
            logger.error(f"Error generating market recap: {str(e)}")
            logger.exception("Full exception details:")
            return None
    
    def generate_tomorrow_outlook(self):
        """Generate outlook for tomorrow's market"""
        logger.info("Starting tomorrow outlook content generation")
        # Get current market data to base prediction on
        market_data = self.crypto_client.get_market_data(count=3)
        
        if not market_data:
            logger.warning("No market data available for tomorrow outlook")
        
        # Prepare data if available
        market_context = ""
        if market_data:
            market_context = "Current market context:\n"
            for coin in market_data:
                market_context += f"- {coin['name']} ({coin['symbol'].upper()}): ${coin['current_price']:.2f}, 24h change: {coin['price_change_percentage_24h']:.2f}%\n"
            
            logger.info(f"Prepared market context for tomorrow outlook with {len(market_data)} coins")
        
        # Create prompt for AI
        prompt = f"""
        Generate a Twitter thread (3 tweets) with an outlook for tomorrow's cryptocurrency market.
        
        {market_context}
        
        Include potential market drivers, events to watch, and general sentiment.
        Make it insightful yet engaging with relevant hashtags like #crypto #MarketOutlook.
        Similar to @aixbt_agent style - professional yet conversational.
        Each tweet should be under 280 characters.
        
        Bot identifier: {self.bot_id}
        """
        
        logger.info("Sending prompt to OpenAI for tomorrow outlook content generation")
        
        # Generate content
        try:
            content = self.ai_client.generate_content(prompt)
            
            if not content:
                logger.error("OpenAI returned empty content for tomorrow outlook")
                return None
            
            logger.info(f"Successfully generated tomorrow outlook content: {len(content)} characters")
            logger.info(f"First 100 chars of tomorrow outlook: {content[:100]}")
            
            # Split into tweets
            tweets = self._format_as_thread(content)
            logger.info(f"Formatted tomorrow outlook into {len(tweets)} tweets")
            
            for i, tweet in enumerate(tweets):
                logger.info(f"Tomorrow Outlook Tweet {i+1}: {tweet[:50]}... ({len(tweet)} chars)")
            
            return tweets
        except Exception as e:
            logger.error(f"Error generating tomorrow outlook: {str(e)}")
            logger.exception("Full exception details:")
            return None
    
    def _format_as_thread(self, content):
        """Format content as a Twitter thread"""
        logger.info("Formatting content into Twitter thread")
        
        if not content:
            logger.warning("Empty content provided to _format_as_thread")
            return []
        
        # Split by double newlines or numbered points
        raw_tweets = []
        
        # Check if content has numbered points (1., 2., etc.)
        if any(line.strip().startswith(str(i) + '.') for i in range(1, 10) for line in content.split('\n')):
            logger.info("Content contains numbered points, splitting by numbered points")
            # Split by numbered points
            current_tweet = ""
            for line in content.split('\n'):
                if any(line.strip().startswith(str(i) + '.') for i in range(1, 10)):
                    if current_tweet:
                        raw_tweets.append(current_tweet.strip())
                    current_tweet = line
                else:
                    current_tweet += '\n' + line
            
            if current_tweet:
                raw_tweets.append(current_tweet.strip())
        else:
            logger.info("Content does not contain numbered points, splitting by paragraphs")
            # Split by paragraphs (double newlines)
            raw_tweets = [tweet.strip() for tweet in content.split('\n\n') if tweet.strip()]
        
        logger.info(f"Split content into {len(raw_tweets)} raw tweets")
        
        # Process each tweet to ensure they're under 280 characters
        processed_tweets = []
        for tweet in raw_tweets:
            # If tweet is too long, split it
            if len(tweet) > 280:
                logger.info(f"Tweet exceeds 280 chars ({len(tweet)} chars), splitting: {tweet[:50]}...")
                # Find a good breaking point (end of sentence)
                sentences = []
                current = ""
                
                for char in tweet:
                    current += char
                    if char in ['.', '!', '?'] and len(current) <= 270:
                        sentences.append(current)
                        current = ""
                
                if current:
                    sentences.append(current)
                
                logger.info(f"Split long tweet into {len(sentences)} sentences")
                processed_tweets.extend(sentences)
            else:
                processed_tweets.append(tweet)
        
        logger.info(f"Finalized thread with {len(processed_tweets)} tweets")
        return processed_tweets

# Helper functions for the main bot
def check_mentions(twitter_client, ai_client):
    """Check and respond to mentions"""
    try:
        logger.info("Checking for new mentions...")
        mentions = twitter_client.get_mentions()
        
        if not mentions:
            logger.info("No new mentions found")
            return
        
        for mention in mentions:
            # Skip if we've already processed this mention
            if mention.id in twitter_client.processed_mentions:
                continue
            
            logger.info(f"Processing mention from @{mention.user.screen_name}: {mention.text}")
            
            # Add to processed set
            twitter_client.processed_mentions.add(mention.id)
            
            # Get the tweet text without the bot's username
            tweet_text = mention.text.replace(f"@{twitter_client.bot_username}", "").strip()
            
            # Generate response
            prompt = f"Generate a helpful response to this tweet about cryptocurrency or AI: '{tweet_text}'. Keep it under 280 characters with a professional yet friendly tone."
            response = ai_client.generate_content(prompt)
            
            if response:
                # Post reply
                twitter_client.post_reply(response, mention.id, mention.user.screen_name)
            else:
                logger.error("Failed to generate response to mention")
        
    except Exception as e:
        logger.error(f"Error checking mentions: {str(e)}")
        logger.exception("Full exception details:")

def check_user_interactions(twitter_client, ai_client=None):
    """Check for user interactions with error handling"""
    try:
        logger.info("Checking for user interactions...")
        if twitter_client.can_read_mentions:
            logger.info("Checking for new mentions...")
            if ai_client:
                check_mentions(twitter_client, ai_client)
            else:
                logger.warning("AI client not provided, skipping mentions processing")
        else:
            logger.info("Skipping mentions processing due to known Twitter API permission issues")
    except Exception as e:
        logger.error(f"Error checking user interactions: {str(e)}")
        logger.exception("Full exception details:")

def run_task(task_name, content_generator, twitter_client, task_func):
    """Run a content generation task with comprehensive error handling"""
    try:
        logger.info(f"Running {task_name} task")
        
        # Generate content
        tweets = task_func()
        
        # Check if we got valid content
        if not tweets or len(tweets) == 0:
            logger.error(f"Failed to generate {task_name} content - skipping post")
            return
        
        # Check if the content appears to be an error message
        error_indicators = ["Unable to generate", "service unavailability", "technical difficulties"]
        if any(indicator in tweets[0] for indicator in error_indicators):
            logger.error(f"Generated content appears to be an error message - skipping post")
            return
        
        # Post thread only if content seems valid
        if twitter_client.can_post_tweets:
            twitter_client.post_thread(tweets)
        else:
            logger.info(f"Generated {task_name} content, but skipping posting due to API restrictions")
            # Log the tweets for verification
            for i, tweet in enumerate(tweets):
                logger.info(f"Would post tweet {i+1}: {tweet[:50]}...")
    except Exception as e:
        logger.error(f"Error executing {task_name} task: {str(e)}")
        logger.exception("Full exception details:")

def main():
    """Main function to run the bot"""
    logger.info("Starting Advanced Twitter Bot...")
    
    # Initialize clients
    rate_limit_mgr = GlobalRateLimitManager()
    twitter_client = TwitterClient(rate_limit_mgr)
    crypto_client = CryptoDataClient()
    news_client = NewsDataClient()
    ai_client = OpenAIHelper()
    content_generator = AIXBTStyleContentGenerator(ai_client, crypto_client, news_client)
    
    # Log initial status
    logger.info("Bot initialized, will start checking mentions in 2 minutes")
    time.sleep(120)  # Initial delay
    
    # First check for user interactions
    check_user_interactions(twitter_client, ai_client)
    
    # Schedule tasks based on available permissions
    task_check_interval = 25  # minutes
    if twitter_client.can_read_mentions:
        schedule.every(task_check_interval).minutes.do(check_mentions, 
                                                     twitter_client=twitter_client, 
                                                     ai_client=ai_client)
    else:
        # Schedule regular checks but with skipping mentions
        schedule.every(task_check_interval).minutes.do(check_user_interactions, 
                                                     twitter_client=twitter_client,
                                                     ai_client=ai_client)
    
    # Schedule content tasks only if we can post tweets
    if twitter_client.can_post_tweets:
        # Schedule content generation tasks
        schedule.every().day.at("09:15").do(run_task, task_name="market update", 
                                          content_generator=content_generator, 
                                          twitter_client=twitter_client,
                                          task_func=content_generator.generate_market_update)
        
        schedule.every().day.at("14:15").do(run_task, task_name="technical analysis", 
                                          content_generator=content_generator, 
                                          twitter_client=twitter_client,
                                          task_func=content_generator.generate_technical_analysis)
        
        schedule.every().day.at("16:45").do(run_task, task_name="crypto news", 
                                          content_generator=content_generator, 
                                          twitter_client=twitter_client,
                                          task_func=content_generator.generate_crypto_news)
        
        schedule.every().day.at("11:45").do(run_task, task_name="AI news", 
                                          content_generator=content_generator, 
                                          twitter_client=twitter_client,
                                          task_func=content_generator.generate_ai_news)
        
        schedule.every().day.at("19:15").do(run_task, task_name="market recap", 
                                          content_generator=content_generator, 
                                          twitter_client=twitter_client,
                                          task_func=content_generator.generate_market_recap)
        
        schedule.every().day.at("21:45").do(run_task, task_name="tomorrow outlook", 
                                          content_generator=content_generator, 
                                          twitter_client=twitter_client,
                                          task_func=content_generator.generate_tomorrow_outlook)
        
        logger.info("Advanced Bot started successfully with optimized scheduled tasks")
    else:
        logger.info("Bot running in monitoring mode only (no tweets will be posted)")
    
    # Main loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.exception("Full exception details:")

if __name__ == "__main__":
    main()