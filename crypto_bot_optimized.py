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
import fcntl
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai_helper import OpenAIHelper

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

# News API key
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# CoinGecko API base URL
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Rate limit file path - shared between bots
RATE_LIMIT_FILE = "/tmp/twitter_rate_limits.json"

# Bot identity ID - different for each bot
BOT_ID = "crypto-bot-primary"

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
        self.can_post_tweets = True  # Flag to track if we have tweet posting permission
        self.can_read_mentions = True  # Flag to track if we have mentions reading permission
        
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
            
            # Check permissions
            try:
                # Test tweet posting (without actually posting)
                self.api_v1.verify_credentials()
            except Exception as e:
                if "403" in str(e):
                    logger.warning("No permission to post tweets")
                    self.can_post_tweets = False
            
            logger.info("Twitter clients set up successfully")
        except Exception as e:
            logger.error(f"Error setting up Twitter clients: {str(e)}")
            raise
    
    def get_mentions(self, max_results=5):
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
                mentions = self.api_v1.mentions_timeline(count=max_results)
                
                if self.rate_limit_manager:
                    self.rate_limit_manager.register_request("mentions")
                
                if mentions:
                    return mentions
                return []
                
            except Exception as e:
                if "429" in str(e):
                    wait_time = (2 ** retry) * 120  # Increased exponential backoff
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
                
                # Use v2 API for tweet posting
                response = self.api_v2.create_tweet(text=text)
                
                if self.rate_limit_manager:
                    self.rate_limit_manager.register_request("tweets")
                
                logger.info(f"Tweet posted successfully via v2 API: {text[:50]}...")
                
                # Create a response object compatible with the rest of the code
                class TweetResponse:
                    def __init__(self, id):
                        self.id = id
                
                return TweetResponse(response.data['id'])
                    
            except Exception as e:
                if "429" in str(e):
                    wait_time = (2 ** retry) * 120  # Increased exponential backoff
                    logger.warning(f"Rate limited. Waiting for {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    retry += 1
                elif "403" in str(e):
                    logger.error(f"Permission error posting tweet: {str(e)}")
                    self.can_post_tweets = False  # Set flag to avoid future tries
                    return None
                else:
                    logger.error(f"Error posting tweet: {str(e)}")
                    return None
        
        logger.error("Max retries reached while posting tweet")
        return None
    
    def post_reply(self, text, tweet_id, username):
        """Post a reply to a tweet with rate limit handling using v2 API"""
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
                    wait_time = 300 + (retry * 300)  # Increased wait time
                    logger.warning(f"Rate limit would be exceeded. Delaying reply for {wait_time} seconds.")
                    time.sleep(wait_time)
                    retry += 1
                    continue
                
                # Use v2 API for reply posting
                response = self.api_v2.create_tweet(
                    text=reply_text,
                    in_reply_to_tweet_id=tweet_id
                )
                
                if self.rate_limit_manager:
                    self.rate_limit_manager.register_request("tweets")
                
                logger.info(f"Reply posted successfully via v2 API: {reply_text[:50]}...")
                
                # Create a response object compatible with the rest of the code
                class TweetResponse:
                    def __init__(self, id):
                        self.id = id
                
                return TweetResponse(response.data['id'])
                    
            except Exception as e:
                if "429" in str(e):
                    wait_time = (2 ** retry) * 120  # Increased exponential backoff
                    logger.warning(f"Rate limited. Waiting for {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    retry += 1
                elif "403" in str(e):
                    logger.error(f"Permission error posting reply: {str(e)}")
                    self.can_post_tweets = False  # Set flag to avoid future tries
                    return None
                else:
                    logger.error(f"Error posting reply: {str(e)}")
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
        
        previous_tweet = None
        
        for tweet_text in tweets_list:
            if previous_tweet is None:
                # First tweet in thread
                response = self.post_tweet(tweet_text)
                if response:
                    previous_tweet = response
                else:
                    logger.error("Failed to post first tweet in thread. Aborting thread.")
                    return
            else:
                # Reply to create thread
                response = self.post_reply(
                    tweet_text, 
                    previous_tweet.id, 
                    self.bot_username
                )
                if response:
                    previous_tweet = response
                else:
                    logger.error("Failed to post tweet in thread. Aborting remaining tweets.")
                    return
            
            # Sleep between tweets to avoid rate limits
            time.sleep(15)  # Increased sleep time between thread tweets
        
        logger.info(f"Successfully posted thread with {len(tweets_list)} tweets")

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
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("CoinGecko API rate limited. Will retry later.")
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
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("CoinGecko API rate limited. Will retry later.")
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
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("CoinGecko API rate limited. Will retry later.")
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
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok" and data.get("totalResults", 0) > 0:
                    return data.get("articles", [])
                return []
            else:
                logger.error(f"Error fetching crypto news: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching crypto news: {str(e)}")
            return []

# Content generator class
class CryptoContentGenerator:
    """Generate cryptocurrency-related content"""
    
    def __init__(self, ai_client, crypto_client, news_client):
        self.ai_client = ai_client
        self.crypto_client = crypto_client
        self.news_client = news_client
        self.bot_id = BOT_ID
    
    def generate_market_update(self):
        """Generate market update content"""
        # Get market data
        market_data = self.crypto_client.get_market_data(count=5)
        global_data = self.crypto_client.get_global_data()
        
        # Check if data was successfully retrieved
        if not market_data or not global_data:
            logger.error("Failed to fetch data for market update")
            return ["Unable to generate market update due to data unavailability. Will retry later. #crypto"]
        
        try:
            # Prepare market summary
            market_summary = {
                "total_market_cap": global_data["data"]["total_market_cap"]["usd"],
                "market_cap_change": global_data["data"]["market_cap_change_percentage_24h_usd"],
                "btc_dominance": global_data["data"]["market_cap_percentage"]["btc"]
            }
            
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
            
            # Create prompt for AI
            prompt = f"""
            Generate a Twitter thread (2-3 tweets) with a cryptocurrency market update.
            Include the following data:
            
            Global Market:
            - Total Market Cap: ${market_summary['total_market_cap']:,.2f}
            - 24h Change: {market_summary['market_cap_change']:.2f}%
            - BTC Dominance: {market_summary['btc_dominance']:.2f}%
            
            Top Performing Coins:
            {', '.join([f"{coin['symbol']}: ${coin['price']:,.2f} ({coin['change_24h']:.2f}%)" for coin in focus_coins])}
            
            Format as engaging tweets with relevant hashtags like #crypto #bitcoin.
            Professional yet conversational style.
            Each tweet should be under 280 characters.
            
            Bot identifier: {self.bot_id}
            """
            
            # Generate content
            content = self.ai_client.generate_content(prompt)
            
            if not content:
                logger.error("Failed to generate market update content")
                return ["Unable to generate market update due to AI service unavailability. #crypto"]
            
            # Split into tweets
            return self._format_as_thread(content)
            
        except Exception as e:
            logger.error(f"Error generating market update: {str(e)}")
            return ["Experiencing technical difficulties with market updates. Will be back soon! #crypto"]
    
    def generate_crypto_news(self):
        """Generate cryptocurrency news update"""
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
        
        # Create prompt for AI
        if news_data:
            prompt = f"""
            Generate a Twitter thread (2-3 tweets) about the latest cryptocurrency news.
            Include these recent news items:
            
            {", ".join([f"'{item['title']}' by {item['source']}" for item in news_data])}
            
            Focus on regulatory updates, notable projects, and industry trends.
            Make it informative yet engaging with relevant hashtags like #crypto #blockchain.
            Professional yet conversational style.
            Each tweet should be under 280 characters.
            
            Bot identifier: {self.bot_id}
            """
        else:
            prompt = f"""
            Generate a Twitter thread (2-3 tweets) about the latest cryptocurrency news.
            Focus on regulatory updates, notable projects, and industry trends.
            
            Make it informative yet engaging with relevant hashtags like #crypto #blockchain.
            Professional yet conversational style.
            Each tweet should be under 280 characters.
            Include one major development and its potential impact.
            
            Bot identifier: {self.bot_id}
            """
        
        # Generate content
        content = self.ai_client.generate_content(prompt)
        
        if not content:
            logger.error("Failed to generate crypto news content")
            return ["Unable to generate cryptocurrency news update due to service unavailability. #crypto"]
        
        # Split into tweets
        return self._format_as_thread(content)
    
    def _format_as_thread(self, content):
        """Format content as a Twitter thread"""
        if not content:
            return []
        
        # Split by double newlines or numbered points
        raw_tweets = []
        
        # Check if content has numbered points (1., 2., etc.)
        if any(line.strip().startswith(str(i) + '.') for i in range(1, 10) for line in content.split('\n')):
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
            # Split by paragraphs (double newlines)
            raw_tweets = [tweet.strip() for tweet in content.split('\n\n') if tweet.strip()]
        
        # Process each tweet to ensure they're under 280 characters
        processed_tweets = []
        for tweet in raw_tweets:
            # If tweet is too long, split it
            if len(tweet) > 280:
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
                
                processed_tweets.extend(sentences)
            else:
                processed_tweets.append(tweet)
        
        return processed_tweets

# User interaction class
class UserInteraction:
    """Handle user interactions via mentions"""
    
    def __init__(self, twitter_client, ai_client, crypto_client):
        self.twitter_client = twitter_client
        self.ai_client = ai_client
        self.crypto_client = crypto_client
        self.processed_mentions = set()
        self.last_check_time = 0
        self.mention_check_interval = 60 * 60  # Check every 60 minutes
    
    def process_mentions(self):
        """Process new mentions and respond appropriately"""
        # Skip mentions processing if we know Twitter API doesn't allow it
        if not self.twitter_client.can_read_mentions:
            logger.info("Skipping mentions processing due to known Twitter API permission issues")
            return
            
        # Check if it's too soon to check again
        current_time = time.time()
        if current_time - self.last_check_time < self.mention_check_interval:
            logger.info(f"Skipping mentions check - next check in {int((self.last_check_time + self.mention_check_interval - current_time) / 60)} minutes")
            return
        
        self.last_check_time = current_time
        logger.info("Checking for new mentions...")
        
        try:
            mentions = self.twitter_client.get_mentions(max_results=5)  # Reduced from 10
            
            if not mentions:
                logger.info("No new mentions found")
                return
            
            # Skip further processing if we can't post tweets
            if not self.twitter_client.can_post_tweets:
                logger.info("Found mentions but skipping replies due to known Twitter API permission issues")
                return
            
            # Process only the most recent mentions if there are many
            mentions = mentions[:3]  # Process at most 3 mentions at a time
            
            for mention in mentions:
                # Skip if already processed
                if mention.id in self.processed_mentions:
                    continue
                
                # Add to processed set
                self.processed_mentions.add(mention.id)
                
                # Process the mention
                self._process_single_mention(mention)
                
                # Sleep to avoid rate limiting
                time.sleep(10)  # Increased from 2 seconds
            
        except Exception as e:
            logger.error(f"Error processing mentions: {str(e)}")
    
    def _process_single_mention(self, mention):
        """Process a single mention and generate response"""
        text = mention.text.lower()
        
        # Determine mention type
        mention_type = self._classify_mention(text)
        
        # Generate response based on mention type
        response = self._generate_response(text, mention_type)
        
        if response:
            # Post reply
            self.twitter_client.post_reply(response, mention.id, mention.user.screen_name)
            logger.info(f"Replied to mention {mention.id}")
        else:
            logger.error(f"Failed to generate response for mention {mention.id}")
    
    def _classify_mention(self, text):
        """Classify the type of mention"""
        text = text.lower()
        
        if any(keyword in text for keyword in ["price", "forecast", "predict", "analysis", "technical"]):
            return "market_analysis"
        elif any(keyword in text for keyword in ["news", "update", "latest"]):
            return "news_request"
        elif any(keyword in text for keyword in ["explain", "what is", "how does", "tutorial"]):
            return "explanation"
        else:
            return "general"
    
    def _generate_response(self, text, mention_type):
        """Generate appropriate response to mention"""
        if mention_type == "market_analysis":
            # Extract coin name if present
            coin = None
            common_coins = ["bitcoin", "btc", "ethereum", "eth", "bnb", "solana", "sol"]
            
            for c in common_coins:
                if c in text:
                    coin = c
                    break
            
            prompt = f"""
            Generate a brief Twitter response (under 280 characters) to analyze the current market status 
            {f'of {coin}' if coin else 'of the crypto market'}.
            
            Make it informative with a professional tone. Include relevant hashtags.
            """
            
        elif mention_type == "news_request":
            prompt = f"""
            Generate a brief Twitter response (under 280 characters) about the latest news in cryptocurrency or AI.
            
            Make it informative with a professional tone. Include relevant hashtags.
            """
            
        elif mention_type == "explanation":
            # Try to identify what needs explaining
            topics = ["blockchain", "cryptocurrency", "bitcoin", "ethereum", "defi", "nft", "ai", "machine learning"]
            topic = None
            
            for t in topics:
                if t in text:
                    topic = t
                    break
            
            prompt = f"""
            Generate a brief Twitter response (under 280 characters) explaining 
            {f'{topic}' if topic else 'the concept mentioned'} in simple terms.
            
            Make it educational with a professional tone. Include relevant hashtags.
            """
            
        else:  # general
            prompt = f"""
            Generate a brief Twitter response (under 280 characters) to this mention:
            "{text}"
            
            Make it engaging with a professional tone. Include relevant hashtags if appropriate.
            """
        
        # Generate content
        content = self.ai_client.generate_content(prompt)
        
        if not content:
            return "Thanks for reaching out! Our AI is currently busy. Please try again later. #cryptocurrency #AI"
        
        # Ensure it's under 280 characters
        if len(content) > 280:
            content = content[:277] + "..."
        
        return content

# Main function for scheduled tasks
def run_scheduled_tasks():
    """Run the bot with scheduled tasks"""
    try:
        # Initialize components
        rate_limit_manager = GlobalRateLimitManager()
        twitter_client = TwitterClient(rate_limit_manager)
        ai_client = OpenAIHelper()  # Using updated OpenAI helper
        crypto_client = CryptoDataClient()
        news_client = NewsDataClient()
        content_generator = CryptoContentGenerator(ai_client, crypto_client, news_client)
        user_interaction = UserInteraction(twitter_client, ai_client, crypto_client)
        
        # Define tasks
        def market_update_task():
            logger.info("Running market update task")
            if not twitter_client.can_post_tweets:
                logger.info("Skipping market update due to Twitter API permission issues")
                return
                
            content = content_generator.generate_market_update()
            if content:
                twitter_client.post_thread(content)
        
        def crypto_news_task():
            logger.info("Running crypto news task")
            if not twitter_client.can_post_tweets:
                logger.info("Skipping crypto news due to Twitter API permission issues")
                return
                
            content = content_generator.generate_crypto_news()
            if content:
                twitter_client.post_thread(content)
        
        def check_mentions_task():
            logger.info("Checking user interactions...")
            user_interaction.process_mentions()
        
        # Optimized scheduling for the crypto bot
        # Active during morning and early afternoon hours
        schedule.every().day.at("05:20").do(market_update_task)
        schedule.every().day.at("07:45").do(crypto_news_task)
        schedule.every().day.at("10:20").do(market_update_task)
        schedule.every().day.at("13:45").do(crypto_news_task)
        
        # Check for mentions every 60 minutes
        schedule.every(60).minutes.do(check_mentions_task)
        
        # Initial check for mentions (with delay to avoid immediate API calls)
        logger.info("Bot initialized, will start checking mentions in 3 minutes")
        time.sleep(180)
        check_mentions_task()
        
        logger.info("Optimized cryptocurrency bot started successfully with optimized scheduled tasks")
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes and retry
    except Exception as e:
        logger.error(f"Error initializing bot: {str(e)}")
        raise

# Run the bot if script is executed directly
if __name__ == "__main__":
    try:
        logger.info("Starting optimized cryptocurrency bot...")
        run_scheduled_tasks()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)