#!/usr/bin/env python3
"""
Enhanced Crypto & AI Twitter Bot
--------------------------------
A comprehensive Twitter bot that posts about cryptocurrency, blockchain, and AI topics
with varied content formats and styles similar to @aixbt_agent.

Features:
- Multi-source data collection (crypto, blockchain, AI news)
- OpenAI-enhanced content generation
- Varied post templates and formatting
- Automatic chart generation
- Rate limit handling with exponential backoff
- Duplicate content prevention
"""

import os
import sys
import json
import time
import random
import logging
import hashlib
import sqlite3
import requests
import tweepy
import feedparser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from datetime import datetime, timedelta
from dotenv import load_dotenv
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API credentials
TWITTER_CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
CRYPTOPANIC_API_KEY = os.environ.get("CRYPTOPANIC_API_KEY")
MESSARI_API_KEY = os.environ.get("MESSARI_API_KEY")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")

# Configure OpenAI API
openai.api_key = OPENAI_API_KEY

# Constants
DATA_DIR = "data"
CHARTS_DIR = "charts"
CACHE_DIR = "cache"
DB_FILE = "enhanced_bot.db"
POSTED_CACHE_FILE = os.path.join(CACHE_DIR, "posted_content.json")
MAX_RETRIES = 5
BACKOFF_FACTOR = 2
POST_CATEGORIES = ["crypto", "ai", "market", "news", "research", "opinion"]
DEFAULT_POST_INTERVAL = 2 * 60 * 60  # 2 hours in seconds

# Ensure directories exist
for directory in [DATA_DIR, CHARTS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize database
def init_db():
    """Initialize SQLite database for content storage and tracking"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create table for storing news items
    c.execute('''
    CREATE TABLE IF NOT EXISTS content_items (
        id INTEGER PRIMARY KEY,
        title TEXT,
        source TEXT,
        url TEXT NOT NULL,
        description TEXT,
        category TEXT,
        subcategory TEXT,
        content TEXT,
        published_date TEXT,
        collected_date TEXT,
        posted INTEGER DEFAULT 0,
        post_date TEXT,
        post_id TEXT,
        importance REAL DEFAULT 0.5,
        hash TEXT UNIQUE
    )
    ''')
    
    # Create table for content templates
    c.execute('''
    CREATE TABLE IF NOT EXISTS templates (
        id INTEGER PRIMARY KEY,
        category TEXT,
        template TEXT,
        last_used TEXT
    )
    ''')
    
    # Create table for API rate limiting
    c.execute('''
    CREATE TABLE IF NOT EXISTS api_limits (
        api_name TEXT PRIMARY KEY,
        last_called TEXT,
        remaining_calls INTEGER,
        reset_time TEXT
    )
    ''')
    
    # Create table for tracking post history
    c.execute('''
    CREATE TABLE IF NOT EXISTS post_history (
        id INTEGER PRIMARY KEY,
        date TEXT,
        category TEXT,
        subcategory TEXT,
        post_id TEXT,
        content TEXT,
        url TEXT,
        media_urls TEXT
    )
    ''')
    
    # Insert default templates if not exists
    default_templates = [
        # Crypto templates
        ("crypto", "🚀 #Crypto | {title}\n\n{description}\n\nSource: {source}\n🔗 {url}", None),
        ("crypto", "💰 #Cryptocurrency Update | {title}\n\n{description}\n\n{url}", None),
        ("crypto", "⚡ #Blockchain News | {title}\n\nVia {source}\n\n{url}", None),
        ("crypto", "📈 Crypto Alert: {title}\n\n{description}\n\n{url}", None),
        ("crypto", "🔗 #Web3 Insight | {title}\n\nFrom {source}\n\n{url}", None),
        
        # AI templates
        ("ai", "🤖 #AI News | {title}\n\n{description}\n\nSource: {source}\n🔗 {url}", None),
        ("ai", "🧠 #ArtificialIntelligence | {title}\n\n{description}\n\n{url}", None),
        ("ai", "💡 AI Development: {title}\n\nVia {source}\n\n{url}", None),
        ("ai", "🔬 #ML Research | {title}\n\n{description}\n\n{url}", None),
        ("ai", "🚀 AI Breakthrough: {title}\n\nFrom {source}\n\n{url}", None),
        ("ai", "📊 #AI Models | {title}\n\n{description}\n\n{url}", None),
        
        # Market templates
        ("market", "📊 Market Update | {title}\n\n{description}\n\n{url} #Crypto #Markets", None),
        ("market", "📉📈 Price Alert | {title}\n\n{description}\n\n{url}", None),
        ("market", "💹 #Trading | {title}\n\n{description}\n\n{url}", None),
        ("market", "⚠️ Market Movement: {title}\n\n{description}\n\n{url}", None),
        
        # Research templates
        ("research", "📑 #Research | {title}\n\n{description}\n\nSource: {source}\n🔗 {url}", None),
        ("research", "🔬 New Study: {title}\n\n{description}\n\n{url}", None),
        ("research", "📊 #AIResearch | {title}\n\nPublished by: {source}\n\n{url}", None),
        
        # News templates 
        ("news", "📰 #News | {title}\n\n{description}\n\nSource: {source}\n🔗 {url}", None),
        ("news", "🔥 Breaking: {title}\n\nVia {source}\n\n{url}", None),
        ("news", "⚡ {title}\n\n{description}\n\nRead more: {url}", None),
        
        # Opinion templates
        ("opinion", "💭 #Opinion | {title}\n\n{description}\n\n{url}", None),
        ("opinion", "🤔 Perspective: {title}\n\nVia {source}\n\n{url}", None)
    ]
    
    for template in default_templates:
        try:
            c.execute("INSERT OR IGNORE INTO templates (category, template, last_used) VALUES (?, ?, ?)", template)
        except sqlite3.Error as e:
            logger.error(f"Error inserting template: {e}")
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

# Load posted content cache
def load_posted_cache():
    """Load cache of previously posted content hashes to prevent duplicates"""
    if os.path.exists(POSTED_CACHE_FILE):
        try:
            with open(POSTED_CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Invalid posted content cache file, creating new one")
    return {"content_hashes": [], "post_ids": []}

# Save posted content cache
def save_posted_cache(cache):
    """Save cache of posted content hashes"""
    with open(POSTED_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# Connect to Twitter API with retry mechanism
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, tweepy.TweepyException))
)
def twitter_api_connect():
    """Connect to Twitter API with OAuth1 authentication and retry mechanism"""
    try:
        client = tweepy.Client(
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
        )
        
        # Also initialize v1 API for media uploads
        auth = tweepy.OAuth1UserHandler(
            TWITTER_CONSUMER_KEY,
            TWITTER_CONSUMER_SECRET,
            TWITTER_ACCESS_TOKEN,
            TWITTER_ACCESS_TOKEN_SECRET
        )
        api_v1 = tweepy.API(auth)
        
        # Test the connection
        client.get_me()
        logger.info("Twitter API connection successful")
        return client, api_v1
    except tweepy.TweepyException as e:
        logger.error(f"Twitter API connection error: {e}")
        raise

# API rate limit management
class APIRateLimitManager:
    """Manage API rate limits to prevent 429 errors"""
    
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
    
    def _get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_file)
    
    def update_rate_limit(self, api_name, remaining_calls, reset_time):
        """Update rate limit information for an API"""
        conn = self._get_connection()
        c = conn.cursor()
        now = datetime.now().isoformat()
        
        try:
            c.execute("""
            INSERT OR REPLACE INTO api_limits 
            (api_name, last_called, remaining_calls, reset_time) 
            VALUES (?, ?, ?, ?)
            """, (api_name, now, remaining_calls, reset_time))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error updating rate limit for {api_name}: {e}")
        finally:
            conn.close()
    
    def can_call_api(self, api_name, min_remaining=5):
        """Check if an API can be called based on rate limit information"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        try:
            c.execute("SELECT * FROM api_limits WHERE api_name = ?", (api_name,))
            limit_info = c.fetchone()
            
            if not limit_info:
                return True
            
            reset_time = datetime.fromisoformat(limit_info["reset_time"]) if limit_info["reset_time"] else None
            now = datetime.now()
            
            # If reset time has passed, we can call the API
            if reset_time and now > reset_time:
                return True
            
            # Check if we have enough remaining calls
            remaining = limit_info["remaining_calls"]
            return remaining > min_remaining
        except sqlite3.Error as e:
            logger.error(f"Error checking rate limit for {api_name}: {e}")
            return False
        finally:
            conn.close()
    
    def wait_for_rate_limit(self, api_name):
        """Wait until rate limit resets for an API"""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        try:
            c.execute("SELECT reset_time FROM api_limits WHERE api_name = ?", (api_name,))
            limit_info = c.fetchone()
            
            if not limit_info or not limit_info["reset_time"]:
                # No rate limit info, wait a default time
                logger.info(f"No rate limit info for {api_name}, waiting 60 seconds")
                time.sleep(60)
                return
            
            reset_time = datetime.fromisoformat(limit_info["reset_time"])
            now = datetime.now()
            
            if now < reset_time:
                wait_seconds = (reset_time - now).total_seconds() + 5  # Add 5 seconds buffer
                logger.info(f"Rate limited for {api_name}, waiting {wait_seconds:.1f} seconds until {reset_time}")
                time.sleep(wait_seconds)
        except sqlite3.Error as e:
            logger.error(f"Error waiting for rate limit for {api_name}: {e}")
            # Default wait of 60 seconds
            time.sleep(60)
        finally:
            conn.close()

# Initialize rate limit manager
rate_limit_manager = APIRateLimitManager()

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_cryptopanic():
    """Fetch cryptocurrency news from CryptoPanic API with retry mechanism"""
    api_name = "cryptopanic"
    
    if not rate_limit_manager.can_call_api(api_name):
        rate_limit_manager.wait_for_rate_limit(api_name)
    
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&currencies=BTC,ETH,SOL,XRP&public=true&kind=news"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Update rate limit info
        remaining = int(response.headers.get("X-RateLimit-Remaining", 50))
        reset_time = datetime.now() + timedelta(seconds=int(response.headers.get("X-RateLimit-Reset", 3600)))
        rate_limit_manager.update_rate_limit(api_name, remaining, reset_time.isoformat())
        
        data = response.json()
        content_items = []
        
        for result in data.get('results', []):
            item = {
                'title': result['title'],
                'source': result['source']['title'],
                'url': result['url'],
                'description': result.get('metadata', {}).get('description', ""),
                'category': 'crypto',
                'subcategory': 'news',
                'published_date': result['created_at'],
                'collected_date': datetime.now().isoformat(),
                'importance': 0.7 if result.get('votes', {}).get('positive', 0) > 5 else 0.5
            }
            content_items.append(item)
        
        logger.info(f"Retrieved {len(content_items)} items from CryptoPanic API")
        return content_items
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from CryptoPanic API: {e}")
        raise

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_messari():
    """Fetch cryptocurrency news and market data from Messari API with retry mechanism"""
    api_name = "messari"
    
    if not rate_limit_manager.can_call_api(api_name):
        rate_limit_manager.wait_for_rate_limit(api_name)
    
    headers = {"x-messari-api-key": MESSARI_API_KEY}
    
    # Get news
    news_url = "https://data.messari.io/api/v1/news"
    
    # Get top assets
    assets_url = "https://data.messari.io/api/v2/assets?fields=name,symbol,metrics/market_data/price_usd,metrics/market_data/percent_change_usd_last_24_hours&limit=20"
    
    content_items = []
    
    try:
        # Fetch news
        news_response = requests.get(news_url, headers=headers, timeout=30)
        news_response.raise_for_status()
        
        # Update rate limit info from news response
        remaining = int(news_response.headers.get("X-RateLimit-Remaining", 50))
        reset_time = datetime.now() + timedelta(seconds=int(news_response.headers.get("X-RateLimit-Reset", 3600)))
        rate_limit_manager.update_rate_limit(api_name, remaining, reset_time.isoformat())
        
        news_data = news_response.json()
        
        for article in news_data.get('data', []):
            item = {
                'title': article['title'],
                'source': 'Messari',
                'url': article['url'],
                'description': article.get('content', ""),
                'category': 'crypto',
                'subcategory': 'news',
                'published_date': article['published_at'],
                'collected_date': datetime.now().isoformat(),
                'importance': 0.6
            }
            content_items.append(item)
        
        # Fetch assets data for market information
        assets_response = requests.get(assets_url, headers=headers, timeout=30)
        assets_response.raise_for_status()
        
        assets_data = assets_response.json()
        
        for asset in assets_data.get('data', []):
            name = asset.get('name', '')
            symbol = asset.get('symbol', '')
            price = asset.get('metrics', {}).get('market_data', {}).get('price_usd', 0)
            change_24h = asset.get('metrics', {}).get('market_data', {}).get('percent_change_usd_last_24_hours', 0)
            
            # Only create market items for significant price movements
            if abs(change_24h) >= 5:  # 5% or more change
                direction = "📈 up" if change_24h > 0 else "📉 down"
                importance = min(0.5 + abs(change_24h) / 100, 0.9)  # Higher importance for larger moves
                
                item = {
                    'title': f"{name} ({symbol}) {direction} {abs(change_24h):.2f}% in the last 24 hours",
                    'source': 'Messari Price Alert',
                    'url': f"https://messari.io/asset/{symbol.lower()}",
                    'description': f"Current price: ${price:.2f} USD",
                    'category': 'crypto',
                    'subcategory': 'market',
                    'published_date': datetime.now().isoformat(),
                    'collected_date': datetime.now().isoformat(),
                    'importance': importance
                }
                content_items.append(item)
        
        logger.info(f"Retrieved {len(content_items)} items from Messari API")
        return content_items
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from Messari API: {e}")
        raise

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_reddit():
    """Fetch posts from relevant subreddits with retry mechanism"""
    api_name = "reddit"
    
    if not rate_limit_manager.can_call_api(api_name):
        rate_limit_manager.wait_for_rate_limit(api_name)
    
    auth_url = "https://www.reddit.com/api/v1/access_token"
    
    # Get Reddit access token
    auth_data = {
        "grant_type": "client_credentials"
    }
    auth_headers = {
        "User-Agent": "Enhanced-Crypto-AI-Twitter-Bot/2.0"
    }
    
    try:
        auth_response = requests.post(
            auth_url,
            auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
            data=auth_data,
            headers=auth_headers,
            timeout=30
        )
        auth_response.raise_for_status()
        
        token_data = auth_response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            logger.error("No Reddit access token received")
            return []
        
        # Subreddits to fetch from, categorized
        subreddits = {
            "crypto": ["cryptocurrency", "bitcoin", "ethereum", "solana", "CryptoMarkets"],
            "ai": ["MachineLearning", "artificial", "singularity", "GPT3", "OpenAI", "StableDiffusion"]
        }
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "Enhanced-Crypto-AI-Twitter-Bot/2.0"
        }
        
        content_items = []
        
        for category, sub_list in subreddits.items():
            for subreddit in sub_list:
                url = f"https://oauth.reddit.com/r/{subreddit}/hot.json?limit=10"
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Update rate limit info
                reset_time = datetime.now() + timedelta(seconds=int(response.headers.get("x-ratelimit-reset", 600)))
                remaining = int(response.headers.get("x-ratelimit-remaining", 50))
                rate_limit_manager.update_rate_limit(api_name, remaining, reset_time.isoformat())
                
                data = response.json()
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    
                    # Skip posts that are pinned, ads, or have low score
                    if post_data.get('pinned') or post_data.get('score', 0) < 50:
                        continue
                    
                    # Calculate importance based on score and comments
                    score = post_data.get('score', 0)
                    num_comments = post_data.get('num_comments', 0)
                    importance = min(0.4 + (score / 1000) + (num_comments / 500), 0.9)
                    
                    item = {
                        'title': post_data.get('title', ''),
                        'source': f"Reddit r/{subreddit}",
                        'url': f"https://www.reddit.com{post_data.get('permalink', '')}",
                        'description': post_data.get('selftext', '')[:150] + "..." if len(post_data.get('selftext', '')) > 150 else post_data.get('selftext', ''),
                        'category': category,
                        'subcategory': 'discussion',
                        'published_date': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                        'collected_date': datetime.now().isoformat(),
                        'importance': importance
                    }
                    content_items.append(item)
                
                # Sleep to avoid rate limits
                time.sleep(2)
        
        logger.info(f"Retrieved {len(content_items)} items from Reddit")
        return content_items
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from Reddit: {e}")
        raise

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_news_api():
    """Fetch news from NewsAPI for both crypto and AI topics with retry mechanism"""
    api_name = "newsapi"
    
    if not rate_limit_manager.can_call_api(api_name):
        rate_limit_manager.wait_for_rate_limit(api_name)
    
    today = datetime.now()
    from_date = (today - timedelta(days=3)).strftime('%Y-%m-%d')
    
    # Multiple queries for different topics
    queries = [
        ("crypto", "blockchain OR cryptocurrency OR bitcoin OR ethereum OR web3"),
        ("ai", "artificial intelligence OR machine learning OR gpt OR llm OR AI models")
    ]
    
    content_items = []
    
    for category, query in queries:
        url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Update rate limit info
            remaining = int(response.headers.get("X-RateLimit-Remaining", 50))
            reset_time = datetime.fromtimestamp(int(response.headers.get("X-RateLimit-Reset", int(time.time()) + 3600)))
            rate_limit_manager.update_rate_limit(api_name, remaining, reset_time.isoformat())
            
            data = response.json()
            
            for article in data.get('articles', [])[:20]:  # Limit to top 20 results
                if article['title'] and article['url']:
                    # Calculate importance based on source and freshness
                    # Newer articles get higher importance
                    pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    days_old = (today - pub_date).days
                    freshness_factor = max(0, 3 - days_old) / 3  # 0-1 scale, newer is higher
                    
                    # Premium sources get higher importance
                    premium_sources = ["Bloomberg", "Financial Times", "Wall Street Journal", "Reuters", "TechCrunch", "MIT Technology Review", "Wired"]
                    source_factor = 0.2 if article['source']['name'] in premium_sources else 0.1
                    
                    importance = min(0.5 + freshness_factor + source_factor, 0.9)
                    
                    item = {
                        'title': article['title'],
                        'source': article['source']['name'],
                        'url': article['url'],
                        'description': article['description'] or "",
                        'category': category,
                        'subcategory': 'news',
                        'published_date': article['publishedAt'],
                        'collected_date': datetime.now().isoformat(),
                        'importance': importance
                    }
                    content_items.append(item)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from News API: {e}")
            raise
    
    logger.info(f"Retrieved {len(content_items)} items from News API")
    return content_items

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_rss_feeds():
    """Fetch news from RSS feeds for both crypto and AI topics with retry mechanism"""
    # List of crypto and AI focused RSS feeds
    feeds = {
        "crypto": [
            "https://cointelegraph.com/rss",
            "https://coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
            "https://cryptoslate.com/feed/",
            "https://blog.coinbase.com/feed"
        ],
        "ai": [
            "https://arxiv.org/rss/cs.AI",
            "https://blog.google/technology/ai/rss/",
            "https://openai.com/blog/rss/",
            "https://blogs.microsoft.com/ai/feed/",
            "https://ai.facebook.com/blog/rss/",
            "https://machinelearning.apple.com/rss.xml"
        ]
    }
    
    content_items = []
    
    for category, feed_list in feeds.items():
        for feed_url in feed_list:
            try:
                feed = feedparser.parse(feed_url)
                source = feed.feed.title if hasattr(feed, 'feed') and hasattr(feed.feed, 'title') else "RSS Feed"
                
                # Determine subcategory based on feed URL
                if category == "crypto":
                    subcategory = "news"
                elif "arxiv" in feed_url:
                    subcategory = "research"
                else:
                    subcategory = "news"
                
                for entry in feed.entries[:5]:  # Get the 5 most recent entries
                    # Skip entries without title or link
                    if not hasattr(entry, 'title') or not hasattr(entry, 'link'):
                        continue
                    
                    # Calculate importance based on source
                    importance = 0.7 if any(premium in feed_url for premium in ["arxiv", "openai", "google", "microsoft"]) else 0.6
                    
                    # Get publication date
                    if hasattr(entry, 'published'):
                        pub_date = entry.published
                    elif hasattr(entry, 'updated'):
                        pub_date = entry.updated
                    else:
                        pub_date = datetime.now().isoformat()
                    
                    # Get description
                    if hasattr(entry, 'summary'):
                        description = entry.summary
                    elif hasattr(entry, 'description'):
                        description = entry.description
                    else:
                        description = ""
                    
                    # Clean up description (remove HTML)
                    # This is a very simple approach, might need improvement
                    description = description.replace('<p>', '').replace('</p>', ' ').replace('<br>', ' ')
                    if len(description) > 200:
                        description = description[:197] + "..."
                    
                    item = {
                        'title': entry.title,
                        'source': source,
                        'url': entry.link,
                        'description': description,
                        'category': category,
                        'subcategory': subcategory,
                        'published_date': pub_date,
                        'collected_date': datetime.now().isoformat(),
                        'importance': importance
                    }
                    content_items.append(item)
                
                logger.info(f"Retrieved {len(feed.entries[:5])} items from {source}")
            except Exception as e:
                logger.error(f"Error parsing feed {feed_url}: {e}")
    
    return content_items

def enhance_with_openai(item, max_retries=3):
    """Generate enhanced descriptions using OpenAI for items with minimal descriptions"""
    if not item['description'] or len(item['description']) < 30:
        # Only generate descriptions for items without good descriptions
        try:
            for attempt in range(max_retries):
                try:
                    # Create prompt based on title and category
                    prompt = f"Write a brief 1-2 sentence expert summary of this {item['category']} news: {item['title']}"
                    
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a crypto and AI expert. Provide concise, informative summaries."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=100,
                        temperature=0.7
                    )
                    
                    summary = response.choices[0].message.content.strip()
                    item['description'] = summary
                    logger.info(f"Enhanced description with OpenAI for: {item['title'][:30]}...")
                    break
                except Exception as e:
                    logger.warning(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait before retry
                    else:
                        raise
        except Exception as e:
            logger.error(f"Failed to enhance with OpenAI after {max_retries} attempts: {e}")
    
    return item

def generate_content_hash(item):
    """Generate a unique hash for content item based on title and URL"""
    hash_input = f"{item['title']}|{item['url']}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def save_content_items(items):
    """Save content items to database with duplicate detection"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    count = 0
    for item in items:
        # Create a unique hash for the item
        item_hash = generate_content_hash(item)
        
        try:
            c.execute('''
            INSERT INTO content_items 
            (title, source, url, description, category, subcategory, published_date, collected_date, importance, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['title'],
                item['source'],
                item['url'],
                item['description'],
                item['category'],
                item['subcategory'],
                item['published_date'],
                item['collected_date'],
                item['importance'],
                item_hash
            ))
            count += 1
        except sqlite3.IntegrityError:
            # Skip duplicates (hash constraint violation)
            pass
    
    conn.commit()
    conn.close()
    logger.info(f"Saved {count} new content items to database")
    return count

def get_unposted_content(category=None, subcategory=None, limit=10, min_importance=0.0):
    """Get unposted content from database with filtering options"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    query = '''
    SELECT * FROM content_items 
    WHERE posted = 0
    '''
    
    params = []
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    if subcategory:
        query += " AND subcategory = ?"
        params.append(subcategory)
    
    if min_importance > 0:
        query += " AND importance >= ?"
        params.append(min_importance)
    
    query += " ORDER BY importance DESC, published_date DESC LIMIT ?"
    params.append(limit)
    
    c.execute(query, params)
    items = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return items

def mark_as_posted(item_id, content, post_id=None):
    """Mark content item as posted in database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Update content item
    post_date = datetime.now().isoformat()
    c.execute("UPDATE content_items SET posted = 1, post_date = ?, post_id = ? WHERE id = ?", 
              (post_date, post_id, item_id))
    
    # Record post in history
    c.execute("SELECT * FROM content_items WHERE id = ?", (item_id,))
    item = c.fetchone()
    
    if item:
        c.execute('''
        INSERT INTO post_history 
        (date, category, subcategory, post_id, content, url) 
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            post_date,
            item[4],  # category
            item[5],  # subcategory
            post_id,
            content,
            item[3]   # url
        ))
    
    conn.commit()
    conn.close()
    logger.info(f"Marked item {item_id} as posted with post ID {post_id}")
    
    # Update posted cache
    cache = load_posted_cache()
    if post_id:
        cache["post_ids"].append(post_id)
    # Limit cache size to last 1000 entries
    if len(cache["post_ids"]) > 1000:
        cache["post_ids"] = cache["post_ids"][-1000:]
    save_posted_cache(cache)

def get_template(category):
    """Get a random template for the given category, prioritizing least recently used"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM templates WHERE category = ? ORDER BY last_used NULLS FIRST, RANDOM() LIMIT 5", (category,))
    templates = [dict(row) for row in c.fetchall()]
    
    if not templates:
        # Fallback to a default template if none found
        conn.close()
        return "{title}\n\n{description}\n\n{url}"
    
    # Choose from the 5 least recently used templates
    template = random.choice(templates)
    
    # Update last_used timestamp
    now = datetime.now().isoformat()
    c.execute("UPDATE templates SET last_used = ? WHERE id = ?", (now, template['id']))
    conn.commit()
    conn.close()
    
    return template['template']

def format_tweet(item):
    """Format tweet content based on item category and template"""
    category = item['category']
    template = get_template(category)
    
    # Truncate description if needed
    desc = item['description'] if item['description'] else ""
    if len(desc) > 100:
        desc = desc[:97] + "..."
    
    # Format the tweet content
    tweet = template.format(
        title=item['title'],
        source=item['source'],
        url=item['url'],
        description=desc
    )
    
    # Ensure tweet is within the 280 character limit
    if len(tweet) > 280:
        # Try to shorten by removing description
        tweet = template.format(
            title=item['title'],
            source=item['source'],
            url=item['url'],
            description=""
        )
        
        # If still too long, truncate title
        if len(tweet) > 280:
            max_title_len = 280 - len(template.format(
                title="",
                source=item['source'],
                url=item['url'],
                description=""
            ))
            
            truncated_title = item['title'][:max_title_len-3] + "..."
            tweet = template.format(
                title=truncated_title,
                source=item['source'],
                url=item['url'],
                description=""
            )
    
    return tweet

def is_duplicate_content(content):
    """Check if content is a duplicate based on similarity to recent posts"""
    cache = load_posted_cache()
    content_hash = hashlib.md5(content.encode()).hexdigest()
    
    if content_hash in cache["content_hashes"]:
        return True
    
    # Add to cache
    cache["content_hashes"].append(content_hash)
    # Limit cache size to last 1000 entries
    if len(cache["content_hashes"]) > 1000:
        cache["content_hashes"] = cache["content_hashes"][-1000:]
    save_posted_cache(cache)
    
    return False

def post_tweet(client, api_v1, content, media_paths=None):
    """Post tweet with optional media attachments and duplicate detection"""
    # Check for duplicate content
    if is_duplicate_content(content):
        logger.warning(f"Duplicate content detected, skipping post: {content[:50]}...")
        return None
    
    try:
        media_ids = []
        
        # Upload media if provided
        if media_paths and api_v1:
            for media_path in media_paths:
                if os.path.exists(media_path):
                    media = api_v1.media_upload(filename=media_path)
                    media_ids.append(media.media_id)
        
        # Post the tweet
        if media_ids:
            tweet = client.create_tweet(text=content, media_ids=media_ids)
        else:
            tweet = client.create_tweet(text=content)
        
        tweet_id = tweet.data['id']
        logger.info(f"Posted tweet (ID: {tweet_id}): {content[:50]}...")
        return tweet_id
    except tweepy.TweepyException as e:
        if "duplicate content" in str(e).lower():
            logger.error(f"Duplicate content error: {e}")
            # Add hash to cache to prevent future attempts
            cache = load_posted_cache()
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash not in cache["content_hashes"]:
                cache["content_hashes"].append(content_hash)
                save_posted_cache(cache)
        else:
            logger.error(f"Error posting tweet: {e}")
        return None

def generate_chart(coin, days=7):
    """Generate price chart for a cryptocurrency"""
    try:
        # Create chart directory if it doesn't exist
        os.makedirs(CHARTS_DIR, exist_ok=True)
        
        # Simple random price data generation for demonstration
        # In a real implementation, fetch actual price data from an API
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()
        
        # Generate synthetic price data with random walk
        start_price = random.uniform(100, 50000) if coin == "BTC" else random.uniform(10, 5000)
        price_volatility = 0.05 if coin == "BTC" else 0.08
        prices = [start_price]
        
        for i in range(1, days):
            change = prices[-1] * random.uniform(-price_volatility, price_volatility)
            prices.append(prices[-1] + change)
        
        # Create a pandas DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices,
        })
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['Price'], color='#1DA1F2', linewidth=2)
        plt.title(f"{coin} Price - Last {days} Days", fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(CHARTS_DIR, f"{coin.lower()}_chart_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(chart_path)
        plt.close()
        
        logger.info(f"Generated chart for {coin} at {chart_path}")
        return chart_path
    except Exception as e:
        logger.error(f"Error generating chart for {coin}: {e}")
        return None

def generate_market_summary():
    """Generate market summary tweet with price data for top coins"""
    try:
        # In a real implementation, fetch actual market data from an API
        top_coins = [
            {"symbol": "BTC", "name": "Bitcoin", "price": random.uniform(20000, 80000), "change_24h": random.uniform(-10, 10)},
            {"symbol": "ETH", "name": "Ethereum", "price": random.uniform(1000, 5000), "change_24h": random.uniform(-10, 10)},
            {"symbol": "BNB", "name": "Binance Coin", "price": random.uniform(200, 800), "change_24h": random.uniform(-10, 10)},
            {"symbol": "SOL", "name": "Solana", "price": random.uniform(50, 300), "change_24h": random.uniform(-10, 10)},
        ]
        
        # Format the summary
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        summary = f"📊 Cryptocurrency Market Update 📈\n🕒 As of {current_time}\n\n"
        
        for coin in top_coins:
            emoji = "🟢" if coin["change_24h"] > 0 else "🔴"
            summary += f"{emoji} {coin['symbol']}: ${coin['price']:.2f} ({coin['change_24h']:.2f}%)\n"
        
        summary += "\n#Crypto #Bitcoin #Ethereum"
        
        return summary
    except Exception as e:
        logger.error(f"Error generating market summary: {e}")
        return None

def generate_ai_research_summary():
    """Generate a summary of recent AI research papers from arXiv"""
    # This would normally fetch data from arXiv API
    # For demonstration, using synthetic data
    try:
        ai_papers = [
            {
                "title": "Advances in Large Language Model Reasoning",
                "authors": "Smith et al.",
                "summary": "This paper explores techniques to improve reasoning capabilities in large language models through novel training approaches.",
                "url": "https://arxiv.org/abs/2403.12345"
            },
            {
                "title": "Efficient Transformer Architecture for Real-time Applications",
                "authors": "Johnson et al.",
                "summary": "A new transformer architecture that reduces computational requirements while maintaining performance for real-time applications.",
                "url": "https://arxiv.org/abs/2403.54321"
            }
        ]
        
        summary = "🔬 Latest #AI Research Highlights 📑\n\n"
        
        for i, paper in enumerate(ai_papers, 1):
            summary += f"{i}. {paper['title']}\n"
            summary += f"   Authors: {paper['authors']}\n"
            summary += f"   {paper['url']}\n\n"
        
        summary += "#MachineLearning #ResearchPapers"
        
        return summary
    except Exception as e:
        logger.error(f"Error generating AI research summary: {e}")
        return None

def fetch_and_store_content():
    """Fetch content from all sources and store in database"""
    try:
        # Initialize database if it doesn't exist
        if not os.path.exists(DB_FILE):
            init_db()
        
        # Fetch from various sources
        content_items = []
        content_items.extend(fetch_cryptopanic())
        content_items.extend(fetch_messari())
        content_items.extend(fetch_news_api())
        content_items.extend(fetch_rss_feeds())
        content_items.extend(fetch_reddit())
        
        # Enhance items with OpenAI (focus on items without good descriptions)
        enhanced_items = []
        for item in content_items:
            if random.random() < 0.3 and (not item['description'] or len(item['description']) < 30):
                enhanced_items.append(enhance_with_openai(item))
            else:
                enhanced_items.append(item)
        
        # Save fetched items to database
        save_count = save_content_items(enhanced_items)
        logger.info(f"Saved {save_count} new content items")
        
        return save_count
    except Exception as e:
        logger.error(f"Error in fetch_and_store_content: {e}")
        return 0

def post_content():
    """Select and post content to Twitter"""
    try:
        # Connect to Twitter API
        client, api_v1 = twitter_api_connect()
        if not client or not api_v1:
            logger.error("Failed to connect to Twitter API")
            return False
        
        # Determine which type of content to post
        # Weight distribution for content types
        content_types = [
            {"name": "regular", "weight": 0.6},  # Regular posts from the database
            {"name": "market", "weight": 0.2},   # Market summaries
            {"name": "research", "weight": 0.1}, # Research summaries
            {"name": "chart", "weight": 0.1}     # Charts and visualizations
        ]
        
        weights = [t["weight"] for t in content_types]
        selected_type = random.choices([t["name"] for t in content_types], weights=weights, k=1)[0]
        
        if selected_type == "regular":
            # Weighted selection of categories
            categories = [
                {"name": "crypto", "weight": 0.4},
                {"name": "ai", "weight": 0.4},
                {"name": "market", "weight": 0.2}
            ]
            
            cat_weights = [c["weight"] for c in categories]
            selected_category = random.choices([c["name"] for c in categories], weights=cat_weights, k=1)[0]
            
            # Get unposted content
            items = get_unposted_content(category=selected_category, limit=5, min_importance=0.5)
            
            if not items:
                # If no items in selected category, try any category
                items = get_unposted_content(limit=5, min_importance=0.5)
            
            if items:
                # Select item based on importance (weighted random)
                importances = [item['importance'] for item in items]
                sum_importance = sum(importances)
                
                if sum_importance > 0:
                    probs = [imp/sum_importance for imp in importances]
                    selected_item = random.choices(items, weights=probs, k=1)[0]
                else:
                    selected_item = random.choice(items)
                
                # Format tweet content
                tweet_content = format_tweet(selected_item)
                
                # Post to Twitter
                tweet_id = post_tweet(client, api_v1, tweet_content)
                
                if tweet_id:
                    # Mark as posted in database
                    mark_as_posted(selected_item['id'], tweet_content, tweet_id)
                    return True
            else:
                logger.info("No unposted content available")
                
        elif selected_type == "market":
            # Generate and post market summary
            market_summary = generate_market_summary()
            if market_summary:
                tweet_id = post_tweet(client, api_v1, market_summary)
                if tweet_id:
                    logger.info(f"Posted market summary with ID {tweet_id}")
                    return True
                
        elif selected_type == "research":
            # Generate and post research summary
            research_summary = generate_ai_research_summary()
            if research_summary:
                tweet_id = post_tweet(client, api_v1, research_summary)
                if tweet_id:
                    logger.info(f"Posted research summary with ID {tweet_id}")
                    return True
                
        elif selected_type == "chart":
            # Generate and post chart
            coin = random.choice(["BTC", "ETH", "SOL"])
            chart_path = generate_chart(coin)
            
            if chart_path:
                tweet_content = f"📊 #{coin} Price Chart - Last 7 Days\n\n#Crypto #Trading #{coin}"
                tweet_id = post_tweet(client, api_v1, tweet_content, [chart_path])
                
                if tweet_id:
                    logger.info(f"Posted chart for {coin} with ID {tweet_id}")
                    return True
        
        return False
    except Exception as e:
        logger.error(f"Error in post_content: {e}")
        return False

def run_bot():
    """Main function to run the bot in continuous mode"""
    logger.info("Starting enhanced crypto & AI Twitter bot")
    
    # Initialize database if it doesn't exist
    if not os.path.exists(DB_FILE):
        init_db()
    
    # Main loop
    while True:
        try:
            # Fetch new content periodically (every 6 hours)
            current_hour = datetime.now().hour
            if current_hour % 6 == 0 and datetime.now().minute < 10:
                logger.info("Fetching new content from all sources")
                fetch_and_store_content()
            
            # Post content
            logger.info("Selecting and posting content")
            success = post_content()
            
            if success:
                logger.info("Successfully posted content")
            else:
                logger.warning("Failed to post content or no suitable content found")
            
            # Random interval between posts (1-3 hours)
            wait_time = random.randint(60 * 60, 3 * 60 * 60)
            logger.info(f"Waiting {wait_time//60//60} hours until next post")
            time.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            # Wait 15 minutes if there's an error
            time.sleep(15 * 60)

def test_run():
    """Run the bot in test mode"""
    logger.info("Running enhanced crypto & AI Twitter bot in test mode")
    
    # Initialize database if it doesn't exist
    if not os.path.exists(DB_FILE):
        init_db()
    
    # Fetch limited content
    logger.info("Fetching content from selected sources")
    
    # Choose one source for testing
    test_source = random.choice(["cryptopanic", "messari", "news", "rss", "reddit"])
    content_items = []
    
    if test_source == "cryptopanic":
        content_items.extend(fetch_cryptopanic())
    elif test_source == "messari":
        content_items.extend(fetch_messari())
    elif test_source == "news":
        content_items.extend(fetch_news_api())
    elif test_source == "rss":
        content_items.extend(fetch_rss_feeds())
    else:
        content_items.extend(fetch_reddit())
    
    # Enhance a few items with OpenAI
    enhanced_items = []
    for item in content_items[:3]:  # Only enhance 3 items for testing
        if not item['description'] or len(item['description']) < 30:
            enhanced_items.append(enhance_with_openai(item))
        else:
            enhanced_items.append(item)
    
    # Save fetched items to database
    save_count = save_content_items(enhanced_items)
    logger.info(f"Saved {save_count} new items")
    
    # Try all content types in test mode
    content_types = ["regular", "market", "research", "chart"]
    
    for content_type in content_types:
        logger.info(f"Testing {content_type} content type")
        
        if content_type == "regular":
            # Get some unposted content
            items = get_unposted_content(limit=3)
            
            if items:
                # Select an item
                selected_item = random.choice(items)
                
                # Format tweet content
                tweet_content = format_tweet(selected_item)
                
                # Log the content without posting
                logger.info(f"TEST MODE - Would post: {tweet_content}")
            else:
                logger.info("No unposted content available")
                
        elif content_type == "market":
            # Generate market summary
            market_summary = generate_market_summary()
            if market_summary:
                logger.info(f"TEST MODE - Would post market summary: {market_summary}")
                
        elif content_type == "research":
            # Generate research summary
            research_summary = generate_ai_research_summary()
            if research_summary:
                logger.info(f"TEST MODE - Would post research summary: {research_summary}")
                
        elif content_type == "chart":
            # Generate chart
            coin = random.choice(["BTC", "ETH", "SOL"])
            chart_path = generate_chart(coin)
            
            if chart_path:
                tweet_content = f"📊 #{coin} Price Chart - Last 7 Days\n\n#Crypto #Trading #{coin}"
                logger.info(f"TEST MODE - Would post chart for {coin} with content: {tweet_content}")
                logger.info(f"Chart saved at: {chart_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running in test mode...")
        test_run()
    else:
        print("Starting enhanced crypto & AI Twitter bot...")
        run_bot()