#!/usr/bin/env python3
"""
Enhanced Crypto & AI Twitter Bot (Final Optimized Version)
---------------------------------------------------------
A comprehensive Twitter bot that focuses on AI content while complementing
an existing cryptocurrency bot, with robust rate limit handling and coordination.

Features:
- Multi-source data collection (crypto, blockchain, AI news)
- OpenAI-enhanced content generation
- Varied post templates and formatting
- Automatic chart generation
- Advanced rate limit handling with exponential backoff
- Duplicate content prevention
- Strict coordination mechanisms to avoid API conflicts
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
import threading
import queue
import socket  # For system-wide locking

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
DATA_DIR = "enhanced_data"
CHARTS_DIR = "enhanced_charts"
CACHE_DIR = "enhanced_cache"
LOCK_DIR = "enhanced_locks"
DB_FILE = "enhanced_bot.db"
POSTED_CACHE_FILE = os.path.join(CACHE_DIR, "posted_content.json")
TWITTER_LOCK_FILE = os.path.join(LOCK_DIR, "twitter_api.lock")
API_LOCK_FILE = os.path.join(LOCK_DIR, "api_access.lock")
RATE_LIMIT_LOG = os.path.join(LOCK_DIR, "rate_limit_log.json")
MAX_RETRIES = 3  # Reduced max retries
BACKOFF_FACTOR = 15  # Increased backoff factor
POST_CATEGORIES = ["crypto", "ai", "market", "news", "research", "opinion"]
DEFAULT_POST_INTERVAL = 10 * 60 * 60  # 10 hours in seconds (further increased to reduce conflicts)
TWITTER_RATE_LIMIT_WINDOW = 15 * 60  # 15 minutes in seconds (Twitter rate limit window)
MAX_TWEETS_PER_WINDOW = 100  # Conservative limit for Twitter API v2
MAX_API_CALLS_SAFETY = 20  # Increased safety margin for API calls

# Global thread-safe queue for Twitter API calls
twitter_api_call_queue = queue.Queue()

# Ensure directories exist
for directory in [DATA_DIR, CHARTS_DIR, CACHE_DIR, LOCK_DIR]:
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
    
    # Create table for Twitter API call tracking
    c.execute('''
    CREATE TABLE IF NOT EXISTS twitter_api_calls (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        endpoint TEXT,
        success INTEGER
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
        
        # AI templates - Extended with more varied formats
        ("ai", "🤖 #AI News | {title}\n\n{description}\n\nSource: {source}\n🔗 {url}", None),
        ("ai", "🧠 #ArtificialIntelligence | {title}\n\n{description}\n\n{url}", None),
        ("ai", "💡 AI Development: {title}\n\nVia {source}\n\n{url}", None),
        ("ai", "🔬 #ML Research | {title}\n\n{description}\n\n{url}", None),
        ("ai", "🚀 AI Breakthrough: {title}\n\nFrom {source}\n\n{url}", None),
        ("ai", "📊 #AI Models | {title}\n\n{description}\n\n{url}", None),
        ("ai", "🧮 #MachineLearning Update | {title}\n\n{description}\n\n{url}", None),
        ("ai", "📱 #AI Tech | {title}\n\nVia {source}\n\n{url}", None),
        ("ai", "⚙️ AI Systems | {title}\n\n{description}\n\n{url}", None),
        ("ai", "🔮 Future of AI: {title}\n\nFrom {source}\n\n{url}", None),
        
        # Market templates
        ("market", "📊 Market Update | {title}\n\n{description}\n\n{url} #Crypto #Markets", None),
        ("market", "📉📈 Price Alert | {title}\n\n{description}\n\n{url}", None),
        ("market", "💹 #Trading | {title}\n\n{description}\n\n{url}", None),
        ("market", "⚠️ Market Movement: {title}\n\n{description}\n\n{url}", None),
        
        # Research templates - Enhanced for academic focus
        ("research", "📑 #Research | {title}\n\n{description}\n\nSource: {source}\n🔗 {url}", None),
        ("research", "🔬 New Study: {title}\n\n{description}\n\n{url}", None),
        ("research", "📊 #AIResearch | {title}\n\nPublished by: {source}\n\n{url}", None),
        ("research", "📘 Academic Paper: {title}\n\nAuthors from {source}\n\n{url}", None),
        ("research", "🧪 Research Findings | {title}\n\n{description}\n\n{url}", None),
        ("research", "🔍 Scientific Discovery: {title}\n\nBy researchers at {source}\n\n{url}", None),
        
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

# System-wide lock implementation using socket
class SystemWideLock:
    """
    Implements system-wide locking to coordinate between different processes
    using a TCP socket. This ensures only one process can access the API at a time.
    """
    def __init__(self, port, timeout=300):
        self.port = port
        self.timeout = timeout
        self.sock = None
        
    def acquire(self):
        """Acquire the system-wide lock"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.bind(('127.0.0.1', self.port))
                logger.info(f"Acquired system-wide lock on port {self.port}")
                return True
            except socket.error:
                # Port is in use, lock is held by another process
                logger.debug(f"Lock on port {self.port} is held by another process, waiting...")
                time.sleep(5)
                
        logger.error(f"Timed out waiting for system-wide lock on port {self.port}")
        return False
        
    def release(self):
        """Release the system-wide lock"""
        if self.sock:
            self.sock.close()
            self.sock = None
            logger.info(f"Released system-wide lock on port {self.port}")

# API lock for coordinating between bots - Port 48451 for Twitter API
twitter_api_lock = SystemWideLock(48451)

# Twitter API rate limit tracking
class TwitterRateLimitManager:
    """Advanced manager for Twitter API rate limits to prevent 429 errors"""
    
    def __init__(self, db_file=DB_FILE, rate_limit_log=RATE_LIMIT_LOG):
        self.db_file = db_file
        self.rate_limit_log = rate_limit_log
        self.lock = threading.Lock()
        self.load_rate_limit_log()
        
    def load_rate_limit_log(self):
        """Load rate limit log from file or create new one"""
        if os.path.exists(self.rate_limit_log):
            try:
                with open(self.rate_limit_log, 'r') as f:
                    self.rate_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.rate_data = {"calls": [], "last_reset": time.time()}
        else:
            self.rate_data = {"calls": [], "last_reset": time.time()}
        
        # Clean up old calls
        self.clean_old_calls()
        self.save_rate_limit_log()
        
    def save_rate_limit_log(self):
        """Save rate limit log to file"""
        with open(self.rate_limit_log, 'w') as f:
            json.dump(self.rate_data, f)
            
    def clean_old_calls(self):
        """Remove calls older than the rate limit window"""
        now = time.time()
        window_start = now - TWITTER_RATE_LIMIT_WINDOW
        self.rate_data["calls"] = [call for call in self.rate_data["calls"] 
                                  if call["timestamp"] > window_start]
        
        # If it's been more than 15 minutes since last reset, reset the count
        if now - self.rate_data["last_reset"] > TWITTER_RATE_LIMIT_WINDOW:
            self.rate_data["last_reset"] = now
            
    def _get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_file)
    
    def record_api_call(self, endpoint, success=True):
        """Record a Twitter API call to track rate limits"""
        with self.lock:
            # Update database
            conn = self._get_connection()
            c = conn.cursor()
            
            try:
                c.execute('''
                INSERT INTO twitter_api_calls 
                (timestamp, endpoint, success) 
                VALUES (?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    endpoint,
                    1 if success else 0
                ))
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Error recording Twitter API call: {e}")
            finally:
                conn.close()
            
            # Update in-memory and file tracking
            self.clean_old_calls()
            self.rate_data["calls"].append({
                "timestamp": time.time(),
                "endpoint": endpoint,
                "success": success
            })
            self.save_rate_limit_log()
    
    def get_calls_in_window(self):
        """Get count of API calls made within the rate limit window"""
        with self.lock:
            self.clean_old_calls()
            return len(self.rate_data["calls"])
    
    def can_make_api_call(self):
        """Check if it's safe to make a Twitter API call based on recent usage"""
        calls_in_window = self.get_calls_in_window()
        max_safe_calls = MAX_TWEETS_PER_WINDOW - MAX_API_CALLS_SAFETY
        
        if calls_in_window >= max_safe_calls:
            logger.warning(f"Twitter API rate limit approaching: {calls_in_window}/{MAX_TWEETS_PER_WINDOW} calls in window")
            return False
        
        return True
    
    def wait_for_rate_limit_reset(self):
        """Wait until enough time has passed that we can safely make API calls again"""
        with self.lock:
            self.clean_old_calls()
            calls_in_window = len(self.rate_data["calls"])
            
            if calls_in_window >= (MAX_TWEETS_PER_WINDOW - MAX_API_CALLS_SAFETY):
                # If we have too many calls, wait until the oldest calls expire
                if self.rate_data["calls"]:
                    oldest_timestamp = min(call["timestamp"] for call in self.rate_data["calls"])
                    now = time.time()
                    time_passed = now - oldest_timestamp
                    time_to_wait = max(0, TWITTER_RATE_LIMIT_WINDOW - time_passed) + 10  # Add 10s buffer
                    
                    if time_to_wait > 0:
                        logger.info(f"Waiting {time_to_wait:.1f} seconds for Twitter rate limit to reset")
                        time.sleep(time_to_wait)
                        self.clean_old_calls()  # Clean again after waiting
                else:
                    # No calls recorded yet, wait a short time as precaution
                    time.sleep(5)

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

# Initialize Twitter rate limit manager
twitter_rate_limit = TwitterRateLimitManager()

# File-based locking mechanism (for systems where socket locking isn't available)
def acquire_twitter_api_lock(max_wait=300):
    """Acquire a lock for Twitter API access to prevent concurrent access from multiple bots"""
    start_time = time.time()
    
    # Try socket-based locking first (preferred method)
    if twitter_api_lock.acquire():
        return True
        
    # Fall back to file-based locking if socket locking fails
    while time.time() - start_time < max_wait:
        # Try to create the lock file
        if not os.path.exists(TWITTER_LOCK_FILE):
            try:
                with open(TWITTER_LOCK_FILE, 'w') as f:
                    f.write(str(os.getpid()))
                return True
            except IOError:
                # Failed to create the file, someone else might be creating it
                time.sleep(1)
                continue
        
        # Lock file exists, check if it's stale (older than 5 minutes)
        try:
            lock_time = os.path.getmtime(TWITTER_LOCK_FILE)
            if time.time() - lock_time > 300:  # 5 minutes
                # Stale lock, remove it
                os.remove(TWITTER_LOCK_FILE)
                continue
            
            # Lock is valid, wait a bit and retry
            time.sleep(5)
        except (IOError, OSError):
            # File might have been removed by another process
            time.sleep(1)
    
    logger.error("Timed out waiting for Twitter API lock")
    return False

def release_twitter_api_lock():
    """Release the Twitter API lock"""
    # Release socket lock if we're using it
    twitter_api_lock.release()
    
    # Also clean up file lock if it exists
    try:
        if os.path.exists(TWITTER_LOCK_FILE):
            with open(TWITTER_LOCK_FILE, 'r') as f:
                pid = f.read().strip()
                
                # Only remove the file if we created it
                if pid == str(os.getpid()):
                    os.remove(TWITTER_LOCK_FILE)
    except (IOError, OSError) as e:
        logger.error(f"Error releasing Twitter API lock: {e}")

# Connect to Twitter API with improved rate limit handling
def is_rate_limit_error(exception):
    """Check if an exception is due to rate limiting"""
    if isinstance(exception, tweepy.TweepyException):
        return "429" in str(exception) or "Too Many Requests" in str(exception)
    return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=15, min=120, max=600),
    retry=retry_if_exception(is_rate_limit_error)
)
def twitter_api_connect():
    """Connect to Twitter API with OAuth1 authentication and advanced retry mechanism"""
    # Check rate limit first
    if not twitter_rate_limit.can_make_api_call():
        twitter_rate_limit.wait_for_rate_limit_reset()
    
    # Acquire lock to ensure we're not competing with the other bot
    if not acquire_twitter_api_lock():
        logger.error("Failed to acquire Twitter API lock")
        raise Exception("Failed to acquire Twitter API lock")
    
    try:
        client = tweepy.Client(
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True  # Let Tweepy handle rate limits too
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
        twitter_rate_limit.record_api_call("get_me", success=True)
        logger.info("Twitter API connection successful")
        return client, api_v1
    except tweepy.TweepyException as e:
        twitter_rate_limit.record_api_call("get_me", success=False)
        logger.error(f"Twitter API connection error: {e}")
        raise
    finally:
        release_twitter_api_lock()

# API rate limit management for non-Twitter APIs
class APIRateLimitManager:
    """Manage API rate limits to prevent 429 errors for third-party APIs"""
    
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

# Initialize rate limit manager for third-party APIs
rate_limit_manager = APIRateLimitManager()

# Data source utilities - prioritizing AI sources
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_ai_research():
    """Fetch AI research papers from ArXiv with retry mechanism"""
    api_name = "arxiv"
    
    categories = [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "stat.ML"
    ]
    
    content_items = []
    
    for category in categories:
        try:
            # ArXiv API base URL (using their RSS feed)
            url = f"http://export.arxiv.org/rss/{category}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the RSS feed
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries[:5]:  # Get the 5 most recent papers
                # Extract author names
                authors = ", ".join([author.name for author in entry.get('authors', [])])
                if not authors:
                    authors = "Multiple Authors"
                
                # Extract categories
                entry_categories = [tag.term for tag in entry.get('tags', [])]
                category_str = ", ".join(entry_categories) if entry_categories else category
                
                # Extract summary and clean it up
                summary = entry.get('summary', '')
                # Clean up summary (basic HTML removal)
                summary = summary.replace('<p>', '').replace('</p>', ' ').replace('<br>', ' ')
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                
                item = {
                    'title': entry.title,
                    'source': 'arXiv',
                    'url': entry.link,
                    'description': f"Authors: {authors}. {summary}",
                    'category': 'ai',
                    'subcategory': 'research',
                    'published_date': entry.get('published', datetime.now().isoformat()),
                    'collected_date': datetime.now().isoformat(),
                    'importance': 0.8  # High importance for research papers
                }
                content_items.append(item)
            
            logger.info(f"Retrieved {len(feed.entries[:5])} AI research papers from arXiv category {category}")
            
            # Sleep to avoid overwhelming the ArXiv API
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Error fetching AI research from arXiv category {category}: {e}")
    
    return content_items

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_ai_blogs():
    """Fetch content from major AI company blogs with retry mechanism"""
    # List of AI company blogs and research sites
    feeds = [
        {"url": "https://blog.google/technology/ai/rss/", "source": "Google AI"},
        {"url": "https://openai.com/blog/rss/", "source": "OpenAI"},
        {"url": "https://blogs.microsoft.com/ai/feed/", "source": "Microsoft AI"},
        {"url": "https://ai.meta.com/blog/rss/", "source": "Meta AI"},
        {"url": "https://machinelearning.apple.com/rss.xml", "source": "Apple ML"},
        {"url": "https://deepmind.google/blog/feed/", "source": "DeepMind"},
        {"url": "https://research.ibm.com/blog/rss.xml", "source": "IBM Research"}
    ]
    
    content_items = []
    
    for feed_info in feeds:
        try:
            feed = feedparser.parse(feed_info["url"])
            source = feed_info["source"]
            
            for entry in feed.entries[:5]:  # Get the 5 most recent entries
                # Skip entries without title or link
                if not hasattr(entry, 'title') or not hasattr(entry, 'link'):
                    continue
                
                # Get description
                if hasattr(entry, 'summary'):
                    description = entry.summary
                elif hasattr(entry, 'description'):
                    description = entry.description
                else:
                    description = ""
                
                # Clean up description (remove HTML)
                description = description.replace('<p>', '').replace('</p>', ' ').replace('<br>', ' ')
                if len(description) > 200:
                    description = description[:197] + "..."
                
                # Get publication date
                if hasattr(entry, 'published'):
                    pub_date = entry.published
                elif hasattr(entry, 'updated'):
                    pub_date = entry.updated
                else:
                    pub_date = datetime.now().isoformat()
                
                item = {
                    'title': entry.title,
                    'source': source,
                    'url': entry.link,
                    'description': description,
                    'category': 'ai',
                    'subcategory': 'news',
                    'published_date': pub_date,
                    'collected_date': datetime.now().isoformat(),
                    'importance': 0.7  # High importance for company blogs
                }
                content_items.append(item)
            
            logger.info(f"Retrieved {len(feed.entries[:5])} items from {source}")
            time.sleep(2)  # Sleep between API calls
            
        except Exception as e:
            logger.error(f"Error parsing feed {feed_info['url']}: {e}")
    
    return content_items

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
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
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
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
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
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
        "User-Agent": "Enhanced-Crypto-AI-Twitter-Bot/3.0"
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
        
        # Subreddits to fetch from, categorized with AI focus
        subreddits = {
            "crypto": ["cryptocurrency", "bitcoin", "ethereum", "solana"],
            "ai": ["MachineLearning", "artificial", "OpenAI", "StableDiffusion", 
                  "GPT3", "LLMs", "ComputerVision", "ArtificialIntelligence", "MLPapers"]
        }
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "Enhanced-Crypto-AI-Twitter-Bot/3.0"
        }
        
        content_items = []
        
        for category, sub_list in subreddits.items():
            for subreddit in sub_list:
                url = f"https://oauth.reddit.com/r/{subreddit}/hot.json?limit=10"
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Update rate limit info - Fixed the parsing issue
                reset_time = datetime.now() + timedelta(seconds=int(float(response.headers.get("x-ratelimit-reset", "600"))))
                remaining = int(float(response.headers.get("x-ratelimit-remaining", "50")))
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
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_news_api():
    """Fetch news from NewsAPI for both crypto and AI topics with retry mechanism"""
    api_name = "newsapi"
    
    if not rate_limit_manager.can_call_api(api_name):
        rate_limit_manager.wait_for_rate_limit(api_name)
    
    today = datetime.now()
    from_date = (today - timedelta(days=3)).strftime('%Y-%m-%d')
    
    # Multiple queries for different topics - prioritize AI
    queries = [
        ("ai", "artificial intelligence OR machine learning OR gpt OR llm OR neural networks OR deep learning"),
        ("ai", "chatgpt OR claude OR gemini OR llama OR stable diffusion OR midjourney OR dalle"),
        ("crypto", "blockchain OR cryptocurrency OR bitcoin OR ethereum OR web3")
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
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
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
    """Post tweet with advanced rate limit handling and optional media attachments"""
    # Acquire Twitter API lock
    if not acquire_twitter_api_lock():
        logger.error("Failed to acquire Twitter API lock")
        return None
    
    try:
        # Check if we can make API calls
        if not twitter_rate_limit.can_make_api_call():
            twitter_rate_limit.wait_for_rate_limit_reset()
        
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
                        twitter_rate_limit.record_api_call("media_upload", success=True)
            
            # Post the tweet
            if media_ids:
                tweet = client.create_tweet(text=content, media_ids=media_ids)
            else:
                tweet = client.create_tweet(text=content)
            
            twitter_rate_limit.record_api_call("create_tweet", success=True)
            
            tweet_id = tweet.data['id']
            logger.info(f"Posted tweet (ID: {tweet_id}): {content[:50]}...")
            return tweet_id
        except tweepy.TweepyException as e:
            twitter_rate_limit.record_api_call("create_tweet", success=False)
            
            if "duplicate content" in str(e).lower():
                logger.error(f"Duplicate content error: {e}")
                # Add hash to cache to prevent future attempts
                cache = load_posted_cache()
                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash not in cache["content_hashes"]:
                    cache["content_hashes"].append(content_hash)
                    save_posted_cache(cache)
            elif is_rate_limit_error(e):
                twitter_rate_limit.wait_for_rate_limit_reset()
                logger.error(f"Rate limit error posting tweet: {e}")
            else:
                logger.error(f"Error posting tweet: {e}")
            return None
    finally:
        release_twitter_api_lock()

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
            },
            {
                "title": "Vision-Language Models for Multimodal Understanding",
                "authors": "Chen et al.",
                "summary": "Exploring the integration of visual and textual information in foundation models.",
                "url": "https://arxiv.org/abs/2403.78901"
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

def generate_ai_news_digest():
    """Generate a digest of recent AI news and developments"""
    try:
        ai_news = [
            {
                "title": "New Large Language Model Released",
                "company": "OpenAI",
                "details": "A new model with improved reasoning and coding capabilities.",
                "url": "https://example.com/ai-news/1"
            },
            {
                "title": "Breakthrough in Computer Vision",
                "company": "Google DeepMind",
                "details": "Novel approach improves object recognition in challenging conditions.",
                "url": "https://example.com/ai-news/2"
            },
            {
                "title": "AI System Achieves State-of-the-Art Results",
                "company": "Anthropic",
                "details": "New benchmark results in reasoning and safety alignment.",
                "url": "https://example.com/ai-news/3"
            }
        ]
        
        summary = "🧠 This Week in #AI - Key Developments 🔍\n\n"
        
        for news in ai_news:
            summary += f"• {news['title']} - {news['company']}\n"
            summary += f"  {news['url']}\n\n"
        
        summary += "#ArtificialIntelligence #TechNews"
        
        return summary
    except Exception as e:
        logger.error(f"Error generating AI news digest: {e}")
        return None

def fetch_and_store_content():
    """Fetch content from all sources and store in database with improved rate limit handling"""
    try:
        # Initialize database if it doesn't exist
        if not os.path.exists(DB_FILE):
            init_db()
        
        # Sleep between API calls to avoid rate limits
        time.sleep(5)
        
        # Fetch from various sources with longer delays between each
        content_items = []
        
        # Prioritize AI content sources
        try:
            logger.info("Fetching AI research papers...")
            content_items.extend(fetch_ai_research())
            time.sleep(15)  # Longer delay between API calls
        except Exception as e:
            logger.error(f"Error fetching AI research: {e}")
            
        try:
            logger.info("Fetching AI blog content...")
            content_items.extend(fetch_ai_blogs())
            time.sleep(15)
        except Exception as e:
            logger.error(f"Error fetching AI blogs: {e}")
        
        # Then fetch crypto sources
        try:
            logger.info("Fetching from CryptoPanic...")
            content_items.extend(fetch_cryptopanic())
            time.sleep(15)
        except Exception as e:
            logger.error(f"Error fetching from CryptoPanic: {e}")
        
        try:
            logger.info("Fetching from Messari...")
            content_items.extend(fetch_messari())
            time.sleep(15)
        except Exception as e:
            logger.error(f"Error fetching from Messari: {e}")
        
        # Fetch from general sources with crypto and AI filtering
        try:
            logger.info("Fetching from News API...")
            content_items.extend(fetch_news_api())
            time.sleep(15)
        except Exception as e:
            logger.error(f"Error fetching from News API: {e}")
        
        try:
            logger.info("Fetching from RSS feeds...")
            content_items.extend(fetch_rss_feeds())
            time.sleep(15)
        except Exception as e:
            logger.error(f"Error fetching from RSS feeds: {e}")
        
        try:
            logger.info("Fetching from Reddit...")
            content_items.extend(fetch_reddit())
        except Exception as e:
            logger.error(f"Error fetching from Reddit: {e}")
        
        # If we have less than 5 items, log a warning
        if len(content_items) < 5:
            logger.warning(f"Retrieved only {len(content_items)} items, which is less than expected")
        
        # Enhance items with OpenAI (focus on items without good descriptions)
        enhanced_items = []
        for item in content_items:
            if random.random() < 0.3 and (not item['description'] or len(item['description']) < 30):
                enhanced_items.append(enhance_with_openai(item))
                time.sleep(3)  # Longer delay between OpenAI calls
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
    """Select and post content to Twitter with advanced rate limit handling"""
    try:
        # Connect to Twitter API
        client, api_v1 = twitter_api_connect()
        if not client or not api_v1:
            logger.error("Failed to connect to Twitter API")
            return False
        
        # Determine which type of content to post - heavily favor AI content
        content_types = [
            {"name": "regular", "weight": 0.5},    # Regular posts from the database
            {"name": "ai_research", "weight": 0.2},# AI research summaries
            {"name": "ai_digest", "weight": 0.2},  # AI news digest
            {"name": "market", "weight": 0.05},    # Market summaries - reduced priority
            {"name": "chart", "weight": 0.05}      # Charts - reduced priority
        ]
        
        weights = [t["weight"] for t in content_types]
        selected_type = random.choices([t["name"] for t in content_types], weights=weights, k=1)[0]
        
        if selected_type == "regular":
            # Weighted selection of categories - Heavily favor AI content
            categories = [
                {"name": "crypto", "weight": 0.1},
                {"name": "ai", "weight": 0.9},  # Much higher weight for AI content
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
                
        elif selected_type == "ai_research":
            # Generate and post AI research summary
            research_summary = generate_ai_research_summary()
            if research_summary:
                tweet_id = post_tweet(client, api_v1, research_summary)
                if tweet_id:
                    logger.info(f"Posted AI research summary with ID {tweet_id}")
                    return True
        
        elif selected_type == "ai_digest":
            # Generate and post AI news digest
            ai_digest = generate_ai_news_digest()
            if ai_digest:
                tweet_id = post_tweet(client, api_v1, ai_digest)
                if tweet_id:
                    logger.info(f"Posted AI news digest with ID {tweet_id}")
                    return True
                
        elif selected_type == "market":
            # Generate and post market summary
            market_summary = generate_market_summary()
            if market_summary:
                tweet_id = post_tweet(client, api_v1, market_summary)
                if tweet_id:
                    logger.info(f"Posted market summary with ID {tweet_id}")
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

def check_posting_schedule():
    """Check if it's time to post based on the enhanced schedule"""
    # Check if we have a record of the last post time
    last_post_file = os.path.join(CACHE_DIR, "last_post_time.txt")
    
    current_time = datetime.now()
    current_hour = current_time.hour
    
    # Enhanced time-based scheduling for better coordination with main bot
    # Post only during particular hours: 8am-10am, 2pm-4pm, 8pm-10pm
    # This creates well-defined posting windows and avoids constant API conflicts
    posting_windows = [
        (8, 10),   # Morning window
        (14, 16),  # Afternoon window 
        (20, 22)   # Evening window
    ]
    
    in_posting_window = False
    for start_hour, end_hour in posting_windows:
        if start_hour <= current_hour < end_hour:
            in_posting_window = True
            break
    
    if not in_posting_window:
        logger.info(f"Current hour {current_hour} is outside posting windows {posting_windows}, skipping")
        return False
    
    # Now check if enough time has passed since the last post
    if os.path.exists(last_post_file):
        try:
            with open(last_post_file, 'r') as f:
                last_post_time = datetime.fromisoformat(f.read().strip())
                time_since_last_post = (current_time - last_post_time).total_seconds()
                
                # Determine if enough time has passed
                # Use a slightly longer interval to reduce frequency
                base_interval = DEFAULT_POST_INTERVAL
                jitter = random.uniform(-0.1 * base_interval, 0.1 * base_interval)
                interval = base_interval + jitter
                
                if time_since_last_post < interval:
                    # Not time to post yet
                    logger.info(f"Last post was {time_since_last_post/3600:.1f} hours ago, waiting until {interval/3600:.1f} hours have passed")
                    return False
        except (ValueError, IOError) as e:
            logger.error(f"Error reading last post time: {e}")
            # If there's an error, continue with posting if we're in a posting window
    
    # Update the last post time
    with open(last_post_file, 'w') as f:
        f.write(current_time.isoformat())
    
    return True

def run_bot():
    """Main function to run the bot in continuous mode with improved rate limit handling"""
    logger.info("Starting enhanced crypto & AI Twitter bot")
    
    # Initialize database if it doesn't exist
    if not os.path.exists(DB_FILE):
        init_db()
    
    # Add significant initial delay to avoid conflicts with main bot
    logger.info("Starting enhanced bot with initial delay to avoid API conflicts")
    time.sleep(45 * 60)  # 45-minute initial delay (increased from 30 minutes)
    
    # Fetch initial content before entering main loop
    logger.info("Performing initial content fetch")
    fetch_and_store_content()
    
    # Main loop
    while True:
        try:
            # Determine current hour
            current_hour = datetime.now().hour
            
            # Fetch new content during non-peak hours (3am, 9am, 3pm, 9pm)
            if current_hour in [3, 9, 15, 21] and datetime.now().minute < 15:
                logger.info("Scheduled content fetch time, collecting new data from sources")
                fetch_and_store_content()
            
            # Check if it's time to post based on enhanced schedule
            if check_posting_schedule():
                # Post content
                logger.info("In posting window, selecting and posting content")
                
                # Wait a random short time before posting to avoid exact pattern
                time.sleep(random.randint(30, 300))  # 30s to 5min random delay
                
                success = post_content()
                
                if success:
                    logger.info("Successfully posted content")
                else:
                    logger.warning("Failed to post content or no suitable content found")
            else:
                logger.info("Not time to post yet, waiting...")
            
            # Sleep for 30 minutes before next check (reduced from 60 minutes to be more responsive)
            sleep_time = 30 * 60
            logger.info(f"Sleeping for {sleep_time/60:.1f} minutes before next check")
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            # Wait 30 minutes if there's an error
            time.sleep(30 * 60)

def test_run():
    """Run the bot in test mode with improved components"""
    logger.info("Running enhanced crypto & AI Twitter bot in test mode")
    
    # Initialize database if it doesn't exist
    if not os.path.exists(DB_FILE):
        init_db()
    
    # Test the Twitter rate limit manager
    logger.info("Testing Twitter rate limit management")
    twitter_rate_limit.record_api_call("test_endpoint", success=True)
    calls_in_window = twitter_rate_limit.get_calls_in_window()
    logger.info(f"Recorded test API call, current calls in window: {calls_in_window}")
    
    # Test the lock mechanism
    logger.info("Testing Twitter API lock mechanism")
    if acquire_twitter_api_lock():
        logger.info("Successfully acquired Twitter API lock")
        release_twitter_api_lock()
        logger.info("Released Twitter API lock")
    else:
        logger.warning("Failed to acquire Twitter API lock")
    
    # Fetch limited content
    logger.info("Fetching content from selected sources")
    
    # Choose one source for testing
    test_source = random.choice(["ai_research", "ai_blogs", "cryptopanic", "messari", "news", "rss", "reddit"])
    content_items = []
    
    if test_source == "ai_research":
        content_items.extend(fetch_ai_research())
    elif test_source == "ai_blogs":
        content_items.extend(fetch_ai_blogs())
    elif test_source == "cryptopanic":
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
    
    # Test scheduling
    logger.info("Testing posting schedule logic")
    should_post = check_posting_schedule()
    logger.info(f"Should post now based on schedule: {should_post}")
    
    # Try all content types in test mode
    content_types = ["regular", "ai_research", "ai_digest", "market", "chart"]
    
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
        
        elif content_type == "ai_research":
            # Generate research summary
            research_summary = generate_ai_research_summary()
            if research_summary:
                logger.info(f"TEST MODE - Would post AI research summary: {research_summary}")
        
        elif content_type == "ai_digest":
            # Generate AI news digest
            ai_digest = generate_ai_news_digest()
            if ai_digest:
                logger.info(f"TEST MODE - Would post AI news digest: {ai_digest}")
                
        elif content_type == "market":
            # Generate market summary
            market_summary = generate_market_summary()
            if market_summary:
                logger.info(f"TEST MODE - Would post market summary: {market_summary}")
                
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
        run_bot()#!/usr/bin/env python3
"""
Enhanced Crypto & AI Twitter Bot (Final Optimized Version)
---------------------------------------------------------
A comprehensive Twitter bot that focuses on AI content while complementing
an existing cryptocurrency bot, with robust rate limit handling and coordination.

Features:
- Multi-source data collection (crypto, blockchain, AI news)
- OpenAI-enhanced content generation
- Varied post templates and formatting
- Automatic chart generation
- Advanced rate limit handling with exponential backoff
- Duplicate content prevention
- Strict coordination mechanisms to avoid API conflicts
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception
import threading
import queue
import socket  # For system-wide locking

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
DATA_DIR = "enhanced_data"
CHARTS_DIR = "enhanced_charts"
CACHE_DIR = "enhanced_cache"
LOCK_DIR = "enhanced_locks"
DB_FILE = "enhanced_bot.db"
POSTED_CACHE_FILE = os.path.join(CACHE_DIR, "posted_content.json")
TWITTER_LOCK_FILE = os.path.join(LOCK_DIR, "twitter_api.lock")
API_LOCK_FILE = os.path.join(LOCK_DIR, "api_access.lock")
RATE_LIMIT_LOG = os.path.join(LOCK_DIR, "rate_limit_log.json")
MAX_RETRIES = 3  # Reduced max retries
BACKOFF_FACTOR = 15  # Increased backoff factor
POST_CATEGORIES = ["crypto", "ai", "market", "news", "research", "opinion"]
DEFAULT_POST_INTERVAL = 10 * 60 * 60  # 10 hours in seconds (further increased to reduce conflicts)
TWITTER_RATE_LIMIT_WINDOW = 15 * 60  # 15 minutes in seconds (Twitter rate limit window)
MAX_TWEETS_PER_WINDOW = 100  # Conservative limit for Twitter API v2
MAX_API_CALLS_SAFETY = 20  # Increased safety margin for API calls

# Global thread-safe queue for Twitter API calls
twitter_api_call_queue = queue.Queue()

# Ensure directories exist
for directory in [DATA_DIR, CHARTS_DIR, CACHE_DIR, LOCK_DIR]:
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
    
    # Create table for Twitter API call tracking
    c.execute('''
    CREATE TABLE IF NOT EXISTS twitter_api_calls (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        endpoint TEXT,
        success INTEGER
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
        
        # AI templates - Extended with more varied formats
        ("ai", "🤖 #AI News | {title}\n\n{description}\n\nSource: {source}\n🔗 {url}", None),
        ("ai", "🧠 #ArtificialIntelligence | {title}\n\n{description}\n\n{url}", None),
        ("ai", "💡 AI Development: {title}\n\nVia {source}\n\n{url}", None),
        ("ai", "🔬 #ML Research | {title}\n\n{description}\n\n{url}", None),
        ("ai", "🚀 AI Breakthrough: {title}\n\nFrom {source}\n\n{url}", None),
        ("ai", "📊 #AI Models | {title}\n\n{description}\n\n{url}", None),
        ("ai", "🧮 #MachineLearning Update | {title}\n\n{description}\n\n{url}", None),
        ("ai", "📱 #AI Tech | {title}\n\nVia {source}\n\n{url}", None),
        ("ai", "⚙️ AI Systems | {title}\n\n{description}\n\n{url}", None),
        ("ai", "🔮 Future of AI: {title}\n\nFrom {source}\n\n{url}", None),
        
        # Market templates
        ("market", "📊 Market Update | {title}\n\n{description}\n\n{url} #Crypto #Markets", None),
        ("market", "📉📈 Price Alert | {title}\n\n{description}\n\n{url}", None),
        ("market", "💹 #Trading | {title}\n\n{description}\n\n{url}", None),
        ("market", "⚠️ Market Movement: {title}\n\n{description}\n\n{url}", None),
        
        # Research templates - Enhanced for academic focus
        ("research", "📑 #Research | {title}\n\n{description}\n\nSource: {source}\n🔗 {url}", None),
        ("research", "🔬 New Study: {title}\n\n{description}\n\n{url}", None),
        ("research", "📊 #AIResearch | {title}\n\nPublished by: {source}\n\n{url}", None),
        ("research", "📘 Academic Paper: {title}\n\nAuthors from {source}\n\n{url}", None),
        ("research", "🧪 Research Findings | {title}\n\n{description}\n\n{url}", None),
        ("research", "🔍 Scientific Discovery: {title}\n\nBy researchers at {source}\n\n{url}", None),
        
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

# System-wide lock implementation using socket
class SystemWideLock:
    """
    Implements system-wide locking to coordinate between different processes
    using a TCP socket. This ensures only one process can access the API at a time.
    """
    def __init__(self, port, timeout=300):
        self.port = port
        self.timeout = timeout
        self.sock = None
        
    def acquire(self):
        """Acquire the system-wide lock"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.bind(('127.0.0.1', self.port))
                logger.info(f"Acquired system-wide lock on port {self.port}")
                return True
            except socket.error:
                # Port is in use, lock is held by another process
                logger.debug(f"Lock on port {self.port} is held by another process, waiting...")
                time.sleep(5)
                
        logger.error(f"Timed out waiting for system-wide lock on port {self.port}")
        return False
        
    def release(self):
        """Release the system-wide lock"""
        if self.sock:
            self.sock.close()
            self.sock = None
            logger.info(f"Released system-wide lock on port {self.port}")

# API lock for coordinating between bots - Port 48451 for Twitter API
twitter_api_lock = SystemWideLock(48451)

# Twitter API rate limit tracking
class TwitterRateLimitManager:
    """Advanced manager for Twitter API rate limits to prevent 429 errors"""
    
    def __init__(self, db_file=DB_FILE, rate_limit_log=RATE_LIMIT_LOG):
        self.db_file = db_file
        self.rate_limit_log = rate_limit_log
        self.lock = threading.Lock()
        self.load_rate_limit_log()
        
    def load_rate_limit_log(self):
        """Load rate limit log from file or create new one"""
        if os.path.exists(self.rate_limit_log):
            try:
                with open(self.rate_limit_log, 'r') as f:
                    self.rate_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.rate_data = {"calls": [], "last_reset": time.time()}
        else:
            self.rate_data = {"calls": [], "last_reset": time.time()}
        
        # Clean up old calls
        self.clean_old_calls()
        self.save_rate_limit_log()
        
    def save_rate_limit_log(self):
        """Save rate limit log to file"""
        with open(self.rate_limit_log, 'w') as f:
            json.dump(self.rate_data, f)
            
    def clean_old_calls(self):
        """Remove calls older than the rate limit window"""
        now = time.time()
        window_start = now - TWITTER_RATE_LIMIT_WINDOW
        self.rate_data["calls"] = [call for call in self.rate_data["calls"] 
                                  if call["timestamp"] > window_start]
        
        # If it's been more than 15 minutes since last reset, reset the count
        if now - self.rate_data["last_reset"] > TWITTER_RATE_LIMIT_WINDOW:
            self.rate_data["last_reset"] = now
            
    def _get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_file)
    
    def record_api_call(self, endpoint, success=True):
        """Record a Twitter API call to track rate limits"""
        with self.lock:
            # Update database
            conn = self._get_connection()
            c = conn.cursor()
            
            try:
                c.execute('''
                INSERT INTO twitter_api_calls 
                (timestamp, endpoint, success) 
                VALUES (?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    endpoint,
                    1 if success else 0
                ))
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Error recording Twitter API call: {e}")
            finally:
                conn.close()
            
            # Update in-memory and file tracking
            self.clean_old_calls()
            self.rate_data["calls"].append({
                "timestamp": time.time(),
                "endpoint": endpoint,
                "success": success
            })
            self.save_rate_limit_log()
    
    def get_calls_in_window(self):
        """Get count of API calls made within the rate limit window"""
        with self.lock:
            self.clean_old_calls()
            return len(self.rate_data["calls"])
    
    def can_make_api_call(self):
        """Check if it's safe to make a Twitter API call based on recent usage"""
        calls_in_window = self.get_calls_in_window()
        max_safe_calls = MAX_TWEETS_PER_WINDOW - MAX_API_CALLS_SAFETY
        
        if calls_in_window >= max_safe_calls:
            logger.warning(f"Twitter API rate limit approaching: {calls_in_window}/{MAX_TWEETS_PER_WINDOW} calls in window")
            return False
        
        return True
    
    def wait_for_rate_limit_reset(self):
        """Wait until enough time has passed that we can safely make API calls again"""
        with self.lock:
            self.clean_old_calls()
            calls_in_window = len(self.rate_data["calls"])
            
            if calls_in_window >= (MAX_TWEETS_PER_WINDOW - MAX_API_CALLS_SAFETY):
                # If we have too many calls, wait until the oldest calls expire
                if self.rate_data["calls"]:
                    oldest_timestamp = min(call["timestamp"] for call in self.rate_data["calls"])
                    now = time.time()
                    time_passed = now - oldest_timestamp
                    time_to_wait = max(0, TWITTER_RATE_LIMIT_WINDOW - time_passed) + 10  # Add 10s buffer
                    
                    if time_to_wait > 0:
                        logger.info(f"Waiting {time_to_wait:.1f} seconds for Twitter rate limit to reset")
                        time.sleep(time_to_wait)
                        self.clean_old_calls()  # Clean again after waiting
                else:
                    # No calls recorded yet, wait a short time as precaution
                    time.sleep(5)

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

# Initialize Twitter rate limit manager
twitter_rate_limit = TwitterRateLimitManager()

# File-based locking mechanism (for systems where socket locking isn't available)
def acquire_twitter_api_lock(max_wait=300):
    """Acquire a lock for Twitter API access to prevent concurrent access from multiple bots"""
    start_time = time.time()
    
    # Try socket-based locking first (preferred method)
    if twitter_api_lock.acquire():
        return True
        
    # Fall back to file-based locking if socket locking fails
    while time.time() - start_time < max_wait:
        # Try to create the lock file
        if not os.path.exists(TWITTER_LOCK_FILE):
            try:
                with open(TWITTER_LOCK_FILE, 'w') as f:
                    f.write(str(os.getpid()))
                return True
            except IOError:
                # Failed to create the file, someone else might be creating it
                time.sleep(1)
                continue
        
        # Lock file exists, check if it's stale (older than 5 minutes)
        try:
            lock_time = os.path.getmtime(TWITTER_LOCK_FILE)
            if time.time() - lock_time > 300:  # 5 minutes
                # Stale lock, remove it
                os.remove(TWITTER_LOCK_FILE)
                continue
            
            # Lock is valid, wait a bit and retry
            time.sleep(5)
        except (IOError, OSError):
            # File might have been removed by another process
            time.sleep(1)
    
    logger.error("Timed out waiting for Twitter API lock")
    return False

def release_twitter_api_lock():
    """Release the Twitter API lock"""
    # Release socket lock if we're using it
    twitter_api_lock.release()
    
    # Also clean up file lock if it exists
    try:
        if os.path.exists(TWITTER_LOCK_FILE):
            with open(TWITTER_LOCK_FILE, 'r') as f:
                pid = f.read().strip()
                
                # Only remove the file if we created it
                if pid == str(os.getpid()):
                    os.remove(TWITTER_LOCK_FILE)
    except (IOError, OSError) as e:
        logger.error(f"Error releasing Twitter API lock: {e}")

# Connect to Twitter API with improved rate limit handling
def is_rate_limit_error(exception):
    """Check if an exception is due to rate limiting"""
    if isinstance(exception, tweepy.TweepyException):
        return "429" in str(exception) or "Too Many Requests" in str(exception)
    return False

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=15, min=120, max=600),
    retry=retry_if_exception(is_rate_limit_error)
)
def twitter_api_connect():
    """Connect to Twitter API with OAuth1 authentication and advanced retry mechanism"""
    # Check rate limit first
    if not twitter_rate_limit.can_make_api_call():
        twitter_rate_limit.wait_for_rate_limit_reset()
    
    # Acquire lock to ensure we're not competing with the other bot
    if not acquire_twitter_api_lock():
        logger.error("Failed to acquire Twitter API lock")
        raise Exception("Failed to acquire Twitter API lock")
    
    try:
        client = tweepy.Client(
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True  # Let Tweepy handle rate limits too
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
        twitter_rate_limit.record_api_call("get_me", success=True)
        logger.info("Twitter API connection successful")
        return client, api_v1
    except tweepy.TweepyException as e:
        twitter_rate_limit.record_api_call("get_me", success=False)
        logger.error(f"Twitter API connection error: {e}")
        raise
    finally:
        release_twitter_api_lock()

# API rate limit management for non-Twitter APIs
class APIRateLimitManager:
    """Manage API rate limits to prevent 429 errors for third-party APIs"""
    
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

# Initialize rate limit manager for third-party APIs
rate_limit_manager = APIRateLimitManager()

# Data source utilities - prioritizing AI sources
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_ai_research():
    """Fetch AI research papers from ArXiv with retry mechanism"""
    api_name = "arxiv"
    
    categories = [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "stat.ML"
    ]
    
    content_items = []
    
    for category in categories:
        try:
            # ArXiv API base URL (using their RSS feed)
            url = f"http://export.arxiv.org/rss/{category}"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the RSS feed
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries[:5]:  # Get the 5 most recent papers
                # Extract author names
                authors = ", ".join([author.name for author in entry.get('authors', [])])
                if not authors:
                    authors = "Multiple Authors"
                
                # Extract categories
                entry_categories = [tag.term for tag in entry.get('tags', [])]
                category_str = ", ".join(entry_categories) if entry_categories else category
                
                # Extract summary and clean it up
                summary = entry.get('summary', '')
                # Clean up summary (basic HTML removal)
                summary = summary.replace('<p>', '').replace('</p>', ' ').replace('<br>', ' ')
                if len(summary) > 200:
                    summary = summary[:197] + "..."
                
                item = {
                    'title': entry.title,
                    'source': 'arXiv',
                    'url': entry.link,
                    'description': f"Authors: {authors}. {summary}",
                    'category': 'ai',
                    'subcategory': 'research',
                    'published_date': entry.get('published', datetime.now().isoformat()),
                    'collected_date': datetime.now().isoformat(),
                    'importance': 0.8  # High importance for research papers
                }
                content_items.append(item)
            
            logger.info(f"Retrieved {len(feed.entries[:5])} AI research papers from arXiv category {category}")
            
            # Sleep to avoid overwhelming the ArXiv API
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Error fetching AI research from arXiv category {category}: {e}")
    
    return content_items

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=60, max=300),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def fetch_ai_blogs():
    """Fetch content from major AI company blogs with retry mechanism"""
    # List of AI company blogs and research sites
    feeds = [
        {"url": "https://blog.google/technology/ai/rss/", "source": "Google AI"},
        {"url": "https://openai.com/blog/rss/", "source": "OpenAI"},
        {"url": "https://blogs.microsoft.com/ai/feed/", "source": "Microsoft AI"},
        {"url": "https://ai.meta.com/blog/rss/", "source": "Meta AI"},
        {"url": "https://machinelearning.apple.com/rss.xml", "source": "Apple ML"},
        {"url": "https://deepmind.google/blog/feed/", "source": "DeepMind"},
        {"url": "https://research.ibm.com/blog/rss.xml", "source": "IBM Research"}
    ]
    
    content_items = []
    
    for feed_info in feeds:
        try:
            feed = feedparser.parse(feed_info["url"])
            source = feed_info["source"]
            
            for entry in feed.entries[:5]:  # Get the 5 most recent entries
                # Skip entries without title or link
                if not hasattr(entry, 'title') or not hasattr(entry, 'link'):
                    continue
                
                # Get description
                if hasattr(entry, 'summary'):
                    description = entry.summary
                elif hasattr(entry, 'description'):
                    description = entry.description
                else:
                    description = ""
                
                # Clean up description (remove HTML)
                description = description.replace('<p>', '').replace('</p>', ' ').replace('<br>', ' ')
                if len(description) > 200:
                    description = description[:197] + "..."
                
                # Get publication date
                if hasattr(entry, 'published'):
                    pub_date = entry.published
                elif hasattr(entry, 'updated'):
                    pub_date = entry.updated
                else:
                    pub_date = datetime.now().isoformat()
                
                item = {
                    'title': entry.title,
                    'source': source,
                    'url': entry.link,
                    'description': description,
                    'category': 'ai',
                    'subcategory': 'news',
                    'published_date': pub_date,
                    'collected_date': datetime.now().isoformat(),
                    'importance': 0.7  # High importance for company blogs
                }
                content_items.append(item)
            
            logger.info(f"Retrieved {len(feed.entries[:5])} items from {source}")
            time.sleep(2)  # Sleep between API