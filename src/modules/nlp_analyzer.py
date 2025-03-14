import logging
import json
import os
import requests
import pandas as pd
import numpy as np
import re
import praw
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class NLPAnalyzer:
    def __init__(self, 
                 reddit_client_id: Optional[str] = None,
                 reddit_client_secret: Optional[str] = None,
                 reddit_user_agent: str = "CryptoAnalysisBot/1.0",
                 cryptocompare_api_key: Optional[str] = None,
                 coingecko_api_key: Optional[str] = None,
                 data_dir: str = "data"):
        """Initialize NLP analyzer with API keys and data directory."""
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.reddit_user_agent = reddit_user_agent
        self.cryptocompare_api_key = cryptocompare_api_key
        self.coingecko_api_key = coingecko_api_key
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize Reddit client if credentials are provided
        self.reddit = None
        if reddit_client_id and reddit_client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
                logger.info("Reddit client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Reddit client: {str(e)}")
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add crypto-specific stop words
        self.stop_words.update([
            'crypto', 'cryptocurrency', 'coin', 'token', 'blockchain',
            'bitcoin', 'btc', 'ethereum', 'eth', 'price', 'market',
            'trading', 'exchange', 'wallet', 'address', 'transaction',
            'buy', 'sell', 'hold', 'hodl', 'moon', 'lambo', 'dip',
            'bullish', 'bearish', 'pump', 'dump', 'fud', 'fomo',
            'ath', 'atl', 'ico', 'airdrop', 'staking', 'mining'
        ])
        
        # Crypto names and symbols mapping (for entity recognition)
        self.crypto_entities = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'xrp': 'ripple',
            'ada': 'cardano',
            'sol': 'solana',
            'doge': 'dogecoin',
            'dot': 'polkadot',
            'avax': 'avalanche',
            'shib': 'shiba inu',
            'ltc': 'litecoin',
            'link': 'chainlink',
            'uni': 'uniswap',
            'matic': 'polygon',
            'cro': 'cronos',
            'atom': 'cosmos',
            'near': 'near protocol',
            'bch': 'bitcoin cash',
            'algo': 'algorand',
            'xlm': 'stellar',
            'etc': 'ethereum classic'
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP analysis."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(filtered_tokens)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract cryptocurrency entities from text."""
        entities = []
        
        # Check for crypto symbols and names
        text_lower = text.lower()
        
        for symbol, name in self.crypto_entities.items():
            pattern_symbol = r'\b' + re.escape(symbol) + r'\b'
            pattern_name = r'\b' + re.escape(name) + r'\b'
            
            if re.search(pattern_symbol, text_lower) or re.search(pattern_name, text_lower):
                entities.append(name)
        
        return entities
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text using keyword frequency."""
        # Preprocess text
        preprocessed_text = self._preprocess_text(text)
        
        if not preprocessed_text:
            return []
        
        # Tokenize and get frequency
        tokens = preprocessed_text.split()
        freq = Counter(tokens)
        
        # Get top 5 keywords
        topics = [word for word, count in freq.most_common(5) if len(word) > 3]
        
        return topics
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using TextBlob."""
        if not text:
            return {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "sentiment": "neutral"
            }
        
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get polarity and subjectivity
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }
    
    def fetch_reddit_posts(self, subreddit_name: str, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetch posts from a specific subreddit."""
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            for post in subreddit.hot(limit=limit):
                # Skip stickied posts
                if post.stickied:
                    continue
                
                # Create post object
                post_obj = {
                    "id": post.id,
                    "title": post.title,
                    "body": post.selftext,
                    "created_utc": datetime.fromtimestamp(post.created_utc).isoformat(),
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "author": str(post.author),
                    "permalink": post.permalink,
                    "url": post.url
                }
                
                # Analyze sentiment
                combined_text = post.title + " " + post.selftext
                post_obj["sentiment"] = self.analyze_sentiment(combined_text)
                
                # Extract entities
                post_obj["entities"] = self._extract_entities(combined_text)
                
                # Extract topics
                post_obj["topics"] = self._extract_topics(combined_text)
                
                posts.append(post_obj)
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
            return posts
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {str(e)}")
            return []
    
    def fetch_crypto_news(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch crypto news from CryptoCompare API."""
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            
            params = {"limit": limit}
            if self.cryptocompare_api_key:
                params["api_key"] = self.cryptocompare_api_key
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Data" not in data:
                logger.error("Failed to fetch crypto news")
                return []
            
            news_articles = []
            for article in data["Data"]:
                # Create article object
                article_obj = {
                    "id": article.get("id"),
                    "title": article.get("title"),
                    "body": article.get("body"),
                    "published_on": datetime.fromtimestamp(article.get("published_on", 0)).isoformat(),
                    "url": article.get("url"),
                    "source": article.get("source"),
                    "categories": article.get("categories", "").split("|")
                }
                
                # Analyze sentiment
                combined_text = article_obj["title"] + " " + article_obj["body"]
                article_obj["sentiment"] = self.analyze_sentiment(combined_text)
                
                # Extract entities
                article_obj["entities"] = self._extract_entities(combined_text)
                
                # Extract topics
                article_obj["topics"] = self._extract_topics(combined_text)
                
                news_articles.append(article_obj)
            
            logger.info(f"Fetched {len(news_articles)} crypto news articles from CryptoCompare")
            return news_articles
        except Exception as e:
            logger.error(f"Failed to fetch crypto news from CryptoCompare API: {str(e)}")
            return []
    
    def fetch_coingecko_news(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch crypto news from CoinGecko API."""
        if not self.coingecko_api_key:
            logger.warning("CoinGecko API key not provided, skipping news fetch")
            return []
        
        try:
            url = "https://pro-api.coingecko.com/api/v3/news"
            
            headers = {
                "x-cg-pro-api-key": self.coingecko_api_key
            }
            
            params = {"per_page": limit}
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or not isinstance(data, list):
                logger.error("Failed to fetch crypto news from CoinGecko Pro API")
                return []
            
            news_articles = []
            for article in data:
                # Create article object
                article_obj = {
                    "id": article.get("news_id"),
                    "title": article.get("title"),
                    "body": article.get("description", ""),
                    "published_on": article.get("published_at"),
                    "url": article.get("url"),
                    "source": article.get("news_site"),
                    "categories": article.get("categories", [])
                }
                
                # Analyze sentiment
                combined_text = article_obj["title"] + " " + article_obj["body"]
                article_obj["sentiment"] = self.analyze_sentiment(combined_text)
                
                # Extract entities
                article_obj["entities"] = self._extract_entities(combined_text)
                
                # Extract topics
                article_obj["topics"] = self._extract_topics(combined_text)
                
                news_articles.append(article_obj)
            
            logger.info(f"Fetched {len(news_articles)} crypto news articles from CoinGecko")
            return news_articles
        except Exception as e:
            logger.error(f"Failed to fetch coin data from CoinGecko: {str(e)}")
            return []
    
    def analyze_entity_sentiment(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze sentiment for each cryptocurrency entity in the data."""
        entity_sentiment = {}
        
        # Process all data
        for item in data:
            # Get entities and sentiment
            entities = item.get("entities", [])
            sentiment_data = item.get("sentiment", {})
            
            if not entities or not sentiment_data:
                continue
            
            polarity = sentiment_data.get("polarity", 0)
            
            # Update sentiment for each entity
            for entity in entities:
                if entity not in entity_sentiment:
                    entity_sentiment[entity] = {
                        "positive": 0,
                        "negative": 0,
                        "neutral": 0,
                        "total": 0,
                        "overall_score": 0.0
                    }
                
                # Update counts
                if polarity > 0.1:
                    entity_sentiment[entity]["positive"] += 1
                elif polarity < -0.1:
                    entity_sentiment[entity]["negative"] += 1
                else:
                    entity_sentiment[entity]["neutral"] += 1
                
                entity_sentiment[entity]["total"] += 1
                
                # Update running average
                current_total = entity_sentiment[entity]["total"]
                current_score = entity_sentiment[entity]["overall_score"]
                entity_sentiment[entity]["overall_score"] = (current_score * (current_total - 1) + polarity) / current_total
        
        # Calculate overall sentiment labels
        for entity, data in entity_sentiment.items():
            if data["overall_score"] > 0.1:
                data["overall"] = "Positive"
            elif data["overall_score"] < -0.1:
                data["overall"] = "Negative"
            else:
                data["overall"] = "Neutral"
        
        return entity_sentiment
    
    def analyze_topic_frequency(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze frequency of topics mentioned in the data."""
        topic_frequency = {}
        
        # Process all data
        for item in data:
            # Get topics
            topics = item.get("topics", [])
            
            if not topics:
                continue
            
            # Update frequency for each topic
            for topic in topics:
                if topic in topic_frequency:
                    topic_frequency[topic] += 1
                else:
                    topic_frequency[topic] = 1
        
        # Sort by frequency
        topic_frequency = dict(sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True))
        
        return topic_frequency
    
    def fetch_cryptopanic_news(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch crypto news from CryptoPanic API."""
        try:
            url = "https://cryptopanic.com/api/v1/posts/"
            
            params = {
                "auth_token": os.getenv('CRYPTOPANIC_API_KEY', ''),  # Optional auth token
                "public": "true",
                "kind": "news",
                "limit": limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "results" not in data:
                logger.error("Failed to fetch news from CryptoPanic")
                return []
            
            news_articles = []
            for article in data["results"]:
                # Create article object
                article_obj = {
                    "id": article.get("id"),
                    "title": article.get("title"),
                    "body": article.get("body", ""),
                    "published_on": article.get("published_at"),
                    "url": article.get("url"),
                    "source": article.get("source", {}).get("title", "CryptoPanic"),
                    "currencies": [c.get("code") for c in article.get("currencies", [])]
                }
                
                # Analyze sentiment
                combined_text = article_obj["title"] + " " + article_obj["body"]
                article_obj["sentiment"] = self.analyze_sentiment(combined_text)
                
                # Extract entities
                article_obj["entities"] = self._extract_entities(combined_text)
                
                # Extract topics
                article_obj["topics"] = self._extract_topics(combined_text)
                
                news_articles.append(article_obj)
            
            logger.info(f"Fetched {len(news_articles)} crypto news articles from CryptoPanic")
            return news_articles
        except Exception as e:
            logger.error(f"Failed to fetch news from CryptoPanic: {str(e)}")
            return []

    def fetch_blockchair_stats(self) -> Dict[str, Any]:
        """Fetch blockchain statistics from Blockchair API."""
        try:
            url = "https://api.blockchair.com/stats"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data:
                logger.error("Failed to fetch stats from Blockchair")
                return {}
            
            logger.info("Successfully fetched blockchain statistics from Blockchair")
            return data["data"]
        except Exception as e:
            logger.error(f"Failed to fetch blockchain statistics: {str(e)}")
            return {}

    def fetch_messari_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch news from Messari API."""
        try:
            url = "https://data.messari.io/api/v1/news"
            
            headers = {}
            if os.getenv('MESSARI_API_KEY'):
                headers["x-messari-api-key"] = os.getenv('MESSARI_API_KEY')
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data:
                logger.error("Failed to fetch news from Messari")
                return []
            
            news_articles = []
            for article in data["data"][:limit]:
                # Create article object
                article_obj = {
                    "id": article.get("id"),
                    "title": article.get("title"),
                    "body": article.get("content", ""),
                    "published_on": article.get("published_at"),
                    "url": article.get("url"),
                    "source": "Messari",
                    "author": article.get("author", {}).get("name")
                }
                
                # Analyze sentiment
                combined_text = article_obj["title"] + " " + article_obj["body"]
                article_obj["sentiment"] = self.analyze_sentiment(combined_text)
                
                # Extract entities
                article_obj["entities"] = self._extract_entities(combined_text)
                
                # Extract topics
                article_obj["topics"] = self._extract_topics(combined_text)
                
                news_articles.append(article_obj)
            
            logger.info(f"Fetched {len(news_articles)} crypto news articles from Messari")
            return news_articles
        except Exception as e:
            logger.error(f"Failed to fetch news from Messari: {str(e)}")
            return []
    
    def collect_social_data(self) -> Dict[str, Any]:
        """Collect and analyze social data from various sources."""
        try:
            social_data = {}
            
            # Fetch Reddit posts
            reddit_posts = []
            subreddits = ["CryptoCurrency", "Bitcoin", "ethereum"]
            
            for subreddit in subreddits:
                posts = self.fetch_reddit_posts(subreddit)
                reddit_posts.extend(posts)
                logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
            
            social_data["reddit_posts"] = reddit_posts
            
            # Fetch news articles from multiple sources
            news_articles = []
            
            # Try CoinGecko first
            coingecko_news = self.fetch_coingecko_news()
            if coingecko_news:
                news_articles.extend(coingecko_news)
            
            # Add CryptoCompare news
            cryptocompare_news = self.fetch_crypto_news()
            if cryptocompare_news:
                news_articles.extend(cryptocompare_news)
                
            # Add CryptoPanic news
            cryptopanic_news = self.fetch_cryptopanic_news()
            if cryptopanic_news:
                news_articles.extend(cryptopanic_news)
                
            # Add Messari news
            messari_news = self.fetch_messari_news()
            if messari_news:
                news_articles.extend(messari_news)
            
            social_data["news"] = news_articles
            
            # Analyze entity sentiment
            all_data = reddit_posts + news_articles
            entity_sentiment = self.analyze_entity_sentiment(all_data)
            social_data["sentiment"] = entity_sentiment
            
            # Analyze topic frequency
            topic_frequency = self.analyze_topic_frequency(all_data)
            social_data["topics"] = topic_frequency
            
            # Add blockchain stats if available
            blockchain_stats = self.fetch_blockchair_stats()
            if blockchain_stats:
                social_data["blockchain_stats"] = blockchain_stats
            
            # Save data to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = os.path.join(self.data_dir, f"social_data_{timestamp}.json")
            
            with open(filename, "w") as f:
                json.dump(social_data, f, default=str)
            
            logger.info(f"Social data collected and saved to {filename}")
            
            return social_data
        except Exception as e:
            logger.error(f"Error collecting social data: {str(e)}")
            return {}