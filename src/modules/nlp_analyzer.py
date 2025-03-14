import os
import nltk
import logging
import requests
import time
import random
import praw
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from dotenv import load_dotenv
import openai
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/nlp_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# This ensures the NLTK data is properly downloaded
def download_nltk_data():
    """Download required NLTK data packages."""
    try:
        # Set download directory to ensure write permissions
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download all necessary NLTK packages
        for package in ['punkt', 'vader_lexicon', 'stopwords']:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"Downloaded NLTK package: {package}")
            except Exception as e:
                logger.error(f"Error downloading NLTK package {package}: {str(e)}")
                
        logger.info("NLTK data download complete")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")
        return False

class NLPAnalyzer:
    def __init__(self):
        # Download required NLTK data
        download_nltk_data()
        
        try:
            self.sia = SentimentIntensityAnalyzer()
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
            self.crypto_terms = self._load_crypto_terms()
            
            # Initialize Reddit client
            self.reddit = self._init_reddit_client()
            
            # Initialize CoinGecko API
            self.coingecko_api_key = os.getenv('COINGECKO_API_KEY', 'CG-HKcah6fnuDW3C4cm1S1c952S')
            
            # Try to initialize OpenAI API (new format)
            try:
                openai.api_key = os.getenv('OPENAI_API_KEY')
                logger.info("OpenAI API key loaded")
            except Exception as e:
                logger.error(f"Error initializing OpenAI API: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error initializing NLP components: {str(e)}")
            # Initialize with empty sets as fallback
            self.stopwords = set()
            self.crypto_terms = self._load_crypto_terms()
    
    def _init_reddit_client(self):
        """Initialize Reddit client using PRAW"""
        try:
            client_id = os.getenv('REDDIT_CLIENT_ID', 'MHi4Wg3lmpnh0OTeM8bnpw')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET', 'oByLtox4x-5nbk2-4z03mWBvivOpRQ')
            user_agent = os.getenv('REDDIT_USER_AGENT', 'danne')
            
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            logger.info("Reddit client initialized")
            return reddit
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {str(e)}")
            return None
        
    def _load_crypto_terms(self):
        """Load a list of common cryptocurrency terms and project names."""
        # You can expand this list or load from a file
        return {
            "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "binance", "bnb",
            "cardano", "ada", "xrp", "dogecoin", "doge", "polkadot", "dot", "avalanche",
            "avax", "chainlink", "link", "litecoin", "ltc", "polygon", "matic", "shiba",
            "defi", "nft", "blockchain", "crypto", "token", "coin", "wallet", "exchange",
            "mining", "staking", "yield", "airdrop", "ico", "altcoin", "bull", "bear",
            "hodl", "fud", "fomo", "dex", "cex", "dao", "web3", "metaverse", "memecoin"
        }
        
    def analyze_sentiment(self, text):
        """Analyze the sentiment of text, returning scores for positive, negative, and compound."""
        try:
            scores = self.sia.polarity_scores(text)
            return scores
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            # Return neutral sentiment as fallback
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    def extract_topics(self, texts):
        """Extract common topics from a list of texts."""
        try:
            all_words = []
            for text in texts:
                try:
                    # Tokenize and clean words - using a simple method if NLTK fails
                    try:
                        words = nltk.word_tokenize(text.lower())
                    except Exception:
                        # Fall back to a simple tokenizer if NLTK fails
                        words = text.lower().split()
                        
                    words = [word for word in words if word.isalpha() and word not in self.stopwords]
                    all_words.extend(words)
                except Exception as e:
                    logger.error(f"Error processing text for topic extraction: {str(e)}")
                    continue
            
            # Count word frequencies and filter for crypto terms
            word_counts = Counter(all_words)
            crypto_topics = {word: count for word, count in word_counts.items() 
                            if word in self.crypto_terms and count > 1}
            
            # Sort by frequency
            sorted_topics = sorted(crypto_topics.items(), key=lambda x: x[1], reverse=True)
            return sorted_topics[:10]  # Return top 10 topics
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            # Return default topics in case of failure
            return [
                ('bitcoin', 5),
                ('ethereum', 4),
                ('defi', 3),
                ('nft', 2),
                ('altcoin', 1)
            ]
    
    def fetch_crypto_news(self, limit=10):
        """Fetch latest cryptocurrency news using proper CoinGecko API endpoints and fallbacks."""
        try:
            # Try CoinGecko Pro API first
            url = "https://pro-api.coingecko.com/api/v3/news"
            headers = {
                'x-cg-pro-api-key': self.coingecko_api_key
            }
            params = {
                'per_page': limit
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            # Check if successful
            if response.status_code == 200:
                news_data = response.json()
                news_items = []
                
                for item in news_data:
                    news_items.append({
                        'title': item.get('title', ''),
                        'body': item.get('description', ''),
                        'url': item.get('url', ''),
                        'source': item.get('author', ''),
                        'published_on': int(datetime.fromisoformat(item.get('published_at', '')).timestamp()) if item.get('published_at') else 0,
                        'categories': item.get('categories', '')
                    })
                
                logger.info(f"Fetched {len(news_items)} crypto news articles from CoinGecko Pro API")
                return news_items
            else:
                logger.error(f"Failed to fetch crypto news from CoinGecko Pro API: {response.status_code}")
                
                # Fall back to public CoinGecko API
                try:
                    url = "https://api.coingecko.com/api/v3/coins/markets"
                    params = {
                        'vs_currency': 'usd',
                        'order': 'market_cap_desc',
                        'per_page': limit,
                        'page': 1,
                        'sparkline': False,
                        'price_change_percentage': '24h'
                    }
                    
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        news_items = []
                        
                        # Convert market data to "news" items
                        for coin in data:
                            change = coin.get('price_change_percentage_24h', 0)
                            direction = "up" if change > 0 else "down"
                            news_items.append({
                                'title': f"{coin.get('name', '')} ({coin.get('symbol', '').upper()}) {direction} {abs(change):.2f}% in 24h",
                                'body': f"Current price: ${coin.get('current_price', 0):,.2f}. Market cap: ${coin.get('market_cap', 0):,.0f}. 24h volume: ${coin.get('total_volume', 0):,.0f}.",
                                'url': f"https://www.coingecko.com/en/coins/{coin.get('id', '')}",
                                'source': 'CoinGecko',
                                'published_on': int(time.time()),
                                'categories': 'Market Data'
                            })
                        
                        logger.info(f"Generated {len(news_items)} market data news items from CoinGecko")
                        return news_items
                    else:
                        logger.error(f"Failed to fetch coin data from CoinGecko: {response.status_code}")
                except Exception as e:
                    logger.error(f"Error with CoinGecko fallback: {str(e)}")
                
                # Fall back to CryptoCompare as last resort
                url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=published_on&limit={}".format(limit)
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    news_data = response.json()
                    news_items = []
                    
                    for item in news_data.get('Data', []):
                        news_items.append({
                            'title': item.get('title', ''),
                            'body': item.get('body', ''),
                            'url': item.get('url', ''),
                            'source': item.get('source', ''),
                            'published_on': item.get('published_on', 0),
                            'categories': item.get('categories', '')
                        })
                    
                    logger.info(f"Fetched {len(news_items)} crypto news articles from CryptoCompare")
                    return news_items
                else:
                    logger.error(f"Failed to fetch crypto news from CryptoCompare: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching crypto news: {str(e)}")
            return []

    def fetch_reddit_sentiment(self, subreddits=['CryptoCurrency', 'Bitcoin', 'ethereum'], limit=25):
        """Fetch posts from crypto-related subreddits using PRAW."""
        all_posts = []
        
        try:
            if not self.reddit:
                logger.error("Reddit client not initialized")
                return []
                
            for subreddit_name in subreddits:
                try:
                    # Get subreddit
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Get hot posts
                    hot_posts = subreddit.hot(limit=limit)
                    
                    for post in hot_posts:
                        if not post.stickied:  # Skip stickied posts
                            all_posts.append({
                                'title': post.title,
                                'text': post.selftext,
                                'subreddit': subreddit_name,
                                'score': post.score,
                                'num_comments': post.num_comments,
                                'created_utc': post.created_utc
                            })
                    
                    logger.info(f"Fetched {len(all_posts)} posts from r/{subreddit_name}")
                except Exception as e:
                    logger.error(f"Error fetching Reddit data from r/{subreddit_name}: {str(e)}")
                
                # Sleep between requests to avoid rate limiting
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in Reddit fetching function: {str(e)}")
        
        # If Reddit fails, try using the public JSON API as fallback
        if not all_posts:
            try:
                logger.warning("PRAW method failed, trying public API as fallback")
                for subreddit in subreddits:
                    try:
                        # Reddit provides a JSON endpoint that doesn't require authentication
                        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
                        
                        # Use a proper user-agent to avoid being blocked
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        
                        # Add a longer timeout
                        response = requests.get(url, headers=headers, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            posts = data.get('data', {}).get('children', [])
                            
                            for post in posts:
                                post_data = post.get('data', {})
                                if not post_data.get('stickied', False):  # Skip stickied posts
                                    all_posts.append({
                                        'title': post_data.get('title', ''),
                                        'text': post_data.get('selftext', ''),
                                        'subreddit': subreddit,
                                        'score': post_data.get('score', 0),
                                        'num_comments': post_data.get('num_comments', 0),
                                        'created_utc': post_data.get('created_utc', 0)
                                    })
                        else:
                            logger.error(f"Failed to fetch posts from r/{subreddit}: {response.status_code}")
                            
                    except Exception as e:
                        logger.error(f"Error fetching Reddit data from r/{subreddit}: {str(e)}")
                        
                    # Sleep between requests to avoid rate limiting
                    time.sleep(3)
            except Exception as e:
                logger.error(f"Error in Reddit fallback function: {str(e)}")
            
        logger.info(f"Fetched {len(all_posts)} posts from Reddit")
        return all_posts
            
    def generate_mock_data(self):
        """Generate mock data when external APIs are unavailable."""
        logger.info("Generating mock crypto news and social data")
        
        # Mock news items
        mock_news = [
            {
                'title': 'Bitcoin Price Surges Past $60,000, Analysts See More Growth',
                'body': 'Bitcoin has seen significant growth this week, passing the $60,000 mark for the first time since last year.',
                'source': 'CryptoNewsSource',
                'published_on': int(time.time()),
                'categories': 'BTC,Market,Price'
            },
            {
                'title': 'Ethereum Upgrade Expected to Improve Network Efficiency',
                'body': 'The upcoming Ethereum protocol upgrade is expected to significantly reduce gas fees and increase transaction throughput.',
                'source': 'BlockchainTimes',
                'published_on': int(time.time()) - 3600,
                'categories': 'ETH,Technology,Upgrade'
            },
            {
                'title': 'DeFi Projects See Record Growth in Total Value Locked',
                'body': 'Decentralized finance platforms have seen a substantial increase in total value locked, indicating growing adoption.',
                'source': 'DeFiPulse',
                'published_on': int(time.time()) - 7200,
                'categories': 'DeFi,Growth,TVL'
            }
        ]
        
        # Mock Reddit posts
        mock_reddit = [
            {
                'title': 'Why I think BTC will hit 100k by EOY - Analysis',
                'text': 'Looking at historical patterns and current market conditions, there are several indicators suggesting Bitcoin could reach $100,000 by the end of the year.',
                'subreddit': 'Bitcoin',
                'score': 423,
                'num_comments': 137,
                'created_utc': int(time.time()) - 12000
            },
            {
                'title': 'Solana ecosystem growth is undervalued - discussion',
                'text': 'Despite recent price action, Solana has seen incredible developer adoption and transaction growth that isn\'t reflected in the current valuation.',
                'subreddit': 'CryptoCurrency',
                'score': 312,
                'num_comments': 89,
                'created_utc': int(time.time()) - 25000
            },
            {
                'title': 'New to Ethereum - what should I know about staking?',
                'text': 'I\'m considering staking my ETH but have concerns about liquidity and the upcoming changes. What are the pros and cons?',
                'subreddit': 'ethereum',
                'score': 156,
                'num_comments': 42,
                'created_utc': int(time.time()) - 36000
            }
        ]
        
        return {
            'mock_news': mock_news,
            'mock_reddit': mock_reddit
        }

    def analyze_alternative_sources(self):
        """Analyze sentiment and extract topics from alternative sources."""
        # Get crypto news
        news_items = self.fetch_crypto_news(limit=20)
        
        # Get Reddit posts
        reddit_posts = self.fetch_reddit_sentiment(limit=30)
        
        # If both sources failed, use mock data
        if not news_items and not reddit_posts:
            logger.warning("No data retrieved from external APIs, using mock data")
            mock_data = self.generate_mock_data()
            news_items = mock_data['mock_news']
            reddit_posts = mock_data['mock_reddit']
        
        # Combine text data for analysis
        all_texts = []
        
        for news in news_items:
            all_texts.append(news.get('title', '') + '. ' + news.get('body', ''))
        
        for post in reddit_posts:
            all_texts.append(post.get('title', '') + '. ' + post.get('text', ''))
        
        # If we still have no texts, generate some basic content
        if not all_texts:
            logger.warning("No text data available, generating basic content")
            all_texts = [
                "Bitcoin shows strong market dominance. Ethereum network continues to grow.",
                "DeFi tokens experiencing high volatility. NFT market remains active with new collections.",
                "Layer 2 solutions gaining adoption. Cross-chain bridges improving interoperability."
            ]
        
        # Calculate sentiment scores
        sentiment_scores = [self.analyze_sentiment(text) for text in all_texts]
        avg_compound = sum(score['compound'] for score in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Extract topics
        topics = self.extract_topics(all_texts)
        
        # If no topics found, provide some basic ones
        if not topics:
            topics = [
                ('bitcoin', 5),
                ('ethereum', 4),
                ('defi', 3),
                ('nft', 2),
                ('altcoin', 1)
            ]
        
        result = {
            'sentiment': avg_compound,
            'topics': topics,
            'news_count': len(news_items),
            'reddit_count': len(reddit_posts),
            'sources': {
                'news': news_items,
                'reddit': reddit_posts
            }
        }
        
        return result
    
    def generate_market_analysis(self, news_articles=None, reddit_posts=None, topics=None, overall_sentiment=0, trading_signals=None, anomalies=None):
        """Generate market analysis using pre-defined templates when OpenAI is not available."""
        try:
            # Try using OpenAI if available
            try:
                # Prepare context for API
                top_topics = ", ".join([f"{topic[0]} ({topic[1]} mentions)" for topic in topics[:5]]) if topics else "No significant topics detected"
                sentiment_text = "bullish" if overall_sentiment > 0.05 else "bearish" if overall_sentiment < -0.05 else "neutral"
                
                # Sample of recent news
                news_sample = []
                if news_articles:
                    for article in news_articles[:3]:
                        news_sample.append(f"Title: {article.get('title', 'No title')}")
                
                news_sample_text = "\n".join(news_sample) if news_sample else "No recent news available"
                
                # Sample of Reddit posts
                reddit_sample = []
                if reddit_posts:
                    for post in reddit_posts[:3]:
                        reddit_sample.append(f"r/{post.get('subreddit', 'Crypto')}: {post.get('title', 'No title')}")
                
                reddit_sample_text = "\n".join(reddit_sample) if reddit_sample else "No recent Reddit posts available"
                
                # Add trading signals info if available
                signals_text = ""
                if trading_signals:
                    signals_list = []
                    for asset, signal in trading_signals.items():
                        if abs(signal.get('strength', 0)) >= 0.6:
                            direction = "BUY" if signal.get('direction') == 'buy' else "SELL"
                            signals_list.append(f"{asset}: {direction} (strength: {signal.get('strength', 0):.2f})")
                    
                    if signals_list:
                        signals_text = "\n\nTrading Signals:\n" + "\n".join(signals_list[:3])
                
                # Add anomalies info if available
                anomalies_text = ""
                if anomalies:
                    anomalies_list = []
                    for anomaly in anomalies:
                        if anomaly.get('confidence', 0) > 60:
                            anomalies_list.append(f"{anomaly.get('asset')}: {anomaly.get('description')} (confidence: {anomaly.get('confidence', 0):.1f}%)")
                    
                    if anomalies_list:
                        anomalies_text = "\n\nMarket Anomalies:\n" + "\n".join(anomalies_list[:3])
                
                prompt = f"""
                Generate a concise crypto market analysis based on the following data:
                
                Top trending topics: {top_topics}
                Overall market sentiment: {sentiment_text} (sentiment score: {overall_sentiment:.2f})
                
                Recent news headlines:
                {news_sample_text}
                
                Recent Reddit discussions:
                {reddit_sample_text}{signals_text}{anomalies_text}
                
                Provide a brief market analysis (2-3 paragraphs) focusing on current trends, potential price movements,
                and actionable insights for traders. Be specific about trending tokens.
                """
                
                # New OpenAI API call format
                client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a crypto market analyst specializing in technical analysis and sentiment-based predictions."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                analysis = response.choices[0].message.content.strip()
                logger.info("Generated market analysis using OpenAI")
                return analysis
            
            except Exception as e:
                logger.error(f"Error using OpenAI: {str(e)}")
                logger.info("Falling back to template-based analysis")
                # If OpenAI fails, fall back to template-based analysis
                
        except Exception as e:
            logger.error(f"Error in market analysis generation: {str(e)}")
            
        # Generate template-based analysis as fallback
        sentiment_text = "bullish" if overall_sentiment > 0.05 else "bearish" if overall_sentiment < -0.05 else "neutral"
        top_topic_names = [topic[0] for topic in topics[:3]] if topics else ["bitcoin", "ethereum", "defi"]
        
        # Include trading signals in template analysis if available
        signals_part = ""
        if trading_signals:
            strong_signals = []
            for asset, signal in trading_signals.items():
                if abs(signal.get('strength', 0)) >= 0.6:
                    direction = "buy" if signal.get('direction') == 'buy' else "sell"
                    strong_signals.append(f"{asset} ({direction})")
            
            if strong_signals:
                signals_part = f" Notable trading signals include {', '.join(strong_signals[:2])}."
        
        # Include anomalies in template analysis if available
        anomalies_part = ""
        if anomalies and len(anomalies) > 0:
            anomalies_part = f" Market anomalies detected in {anomalies[0].get('asset', 'some assets')} warrant caution."
        
        # Templates for market analysis
        bullish_templates = [
            f"The crypto market is showing {sentiment_text} momentum today, with {top_topic_names[0]} leading the way. Technical indicators suggest a potential continuation of the upward trend. {top_topic_names[1]} and {top_topic_names[2]} are also showing promising signals, with increased trading volumes and positive community sentiment.{signals_part}{anomalies_part}",
            
            f"Market sentiment is {sentiment_text} as {top_topic_names[0]} breaks key resistance levels. Traders should watch the $60,000 level for Bitcoin as it remains crucial for the overall market direction. {top_topic_names[1]} projects are seeing increased adoption, potentially offering good entry points for medium-term positions.{signals_part}{anomalies_part}",
            
            f"Recent developments in the {top_topic_names[0]} ecosystem are driving positive momentum across the market. The overall {sentiment_text} sentiment is reinforced by institutional interest and favorable regulatory news. Consider allocating to {top_topic_names[1]} and {top_topic_names[2]} for diversified exposure to this growth trend.{signals_part}{anomalies_part}"
        ]
        
        bearish_templates = [
            f"The crypto market is experiencing a {sentiment_text} phase, with {top_topic_names[0]} showing signs of consolidation. Key support levels should be monitored closely. {top_topic_names[1]} and {top_topic_names[2]} might face short-term pressure, though long-term fundamentals remain intact.{signals_part}{anomalies_part}",
            
            f"Market sentiment has turned {sentiment_text} as {top_topic_names[0]} tests critical support levels. Risk management should be prioritized in the current environment. {top_topic_names[1]} projects could see volatility ahead, though this may present buying opportunities for patient investors.{signals_part}{anomalies_part}",
            
            f"Caution is advised as the market displays {sentiment_text} signals. {top_topic_names[0]} is approaching key technical levels that could determine the next major move. Consider reducing exposure to high-beta assets and focus on blue-chip projects like Bitcoin and Ethereum until market direction becomes clearer.{signals_part}{anomalies_part}"
        ]
        
        neutral_templates = [
            f"The crypto market is showing {sentiment_text} behavior with sideways movement for {top_topic_names[0]}. This consolidation phase might precede a significant move in either direction. {top_topic_names[1]} and {top_topic_names[2]} are worth monitoring for early signals of the next trend.{signals_part}{anomalies_part}",
            
            f"Market sentiment remains {sentiment_text} as traders await catalysts. {top_topic_names[0]} is trading within a defined range, suggesting accumulation. {top_topic_names[1]} developments could provide insight into emerging trends worth following.{signals_part}{anomalies_part}",
            
            f"The current {sentiment_text} market environment presents both opportunities and risks. {top_topic_names[0]} is holding key levels while {top_topic_names[1]} shows promising fundamentals despite price uncertainty. Dollar-cost averaging might be an appropriate strategy in this range-bound market.{signals_part}{anomalies_part}"
        ]
        
        # Select template based on sentiment
        templates = bullish_templates if overall_sentiment > 0.05 else bearish_templates if overall_sentiment < -0.05 else neutral_templates
        analysis = random.choice(templates)
        
        logger.info("Generated market analysis using templates")
        return analysis
    
    def generate_response_to_query(self, query, context=None):
        """Generate responses to user queries using templates when OpenAI is not available."""
        try:
            # Try using OpenAI if available
            try:
                system_prompt = """
                You are a crypto market analysis AI assistant. Respond to the user's query with accurate, 
                helpful information about cryptocurrency markets, tokens, or trends. Be concise but informative.
                Your tone should be slightly casual but professional, and include specific data points when available.
                If you don't know something, be honest about limitations rather than making up information.
                """
                
                user_prompt = query
                if context:
                    user_prompt = f"Context: {context}\n\nUser query: {query}"
                
                # New OpenAI API call format
                client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                reply = response.choices[0].message.content.strip()
                logger.info(f"Generated response to query: {query[:50]}...")
                return reply
            
            except Exception as e:
                logger.error(f"Error using OpenAI: {str(e)}")
                logger.info("Falling back to template-based response")
        
        except Exception as e:
            logger.error(f"Error in query response generation: {str(e)}")
        
        # Generate template-based response as fallback
        # Extract potentially relevant keywords from the query
        query_lower = query.lower()
        
        # Check if the query is about specific cryptocurrencies
        crypto_mentioned = False
        mentioned_crypto = None
        for crypto in self.crypto_terms:
            if crypto in query_lower:
                crypto_mentioned = True
                mentioned_crypto = crypto
                break
        
        # Check query intent
        if "price" in query_lower or "prediction" in query_lower or "forecast" in query_lower:
            if crypto_mentioned and mentioned_crypto:
                templates = [
                    f"{mentioned_crypto.title()} has been showing interesting price action lately. While I can't predict exact prices, technical indicators suggest watching key support and resistance levels. Always do your own research before making investment decisions.",
                    f"Regarding {mentioned_crypto.title()}, the market has been dynamic with various factors affecting prices. Focus on fundamentals and market sentiment rather than short-term predictions for better long-term results.",
                    f"The {mentioned_crypto.title()} market has seen volatility recently. Consider looking at on-chain metrics and development activity for deeper insights beyond price action."
                ]
            else:
                templates = [
                    "The crypto market remains dynamic with various factors influencing prices. It's important to consider fundamentals, market sentiment, and technical indicators rather than focusing solely on price predictions.",
                    "Predicting exact prices is challenging due to market volatility. Consider dollar-cost averaging and proper risk management strategies instead of trying to time the market perfectly.",
                    "Market analysis suggests watching key levels for Bitcoin as it often influences the broader market. Always conduct thorough research and consider your risk tolerance before making investment decisions."
                ]
        elif "analysis" in query_lower or "opinion" in query_lower or "thoughts" in query_lower:
            if crypto_mentioned and mentioned_crypto:
                templates = [
                    f"Based on recent developments, {mentioned_crypto.title()} shows promising technical indicators while maintaining strong fundamentals. Community growth and development activity remain key metrics to watch.",
                    f"My analysis of {mentioned_crypto.title()} focuses on both on-chain metrics and market sentiment. Recent network activity suggests continued adoption despite market fluctuations.",
                    f"Looking at {mentioned_crypto.title()}, there are several factors worth considering: development progress, adoption metrics, and overall market correlation all play important roles in its performance."
                ]
            else:
                templates = [
                    "The current market displays mixed signals with some assets showing strength while others consolidate. Risk management remains crucial in this environment.",
                    "Market analysis suggests we're in a critical phase where broader economic factors are influencing crypto alongside internal market dynamics. Diversification is particularly important now.",
                    "My thoughts on the market center on the importance of fundamentals during uncertain periods. Projects with strong development activity and real-world utility tend to outperform in the long run."
                ]
        elif "market" in query_lower:
            templates = [
                "The crypto market is currently navigating through various macro factors. Bitcoin's performance remains a key indicator for the broader market direction.",
                "Market conditions show interesting patterns with DeFi and Layer 2 solutions seeing increased activity. This sector rotation is typical during consolidation phases.",
                "The overall market structure suggests preparing for potential volatility ahead. Consider reviewing your portfolio allocation and risk management strategies."
            ]
        else:
            templates = [
                "I appreciate your question about the crypto markets. For the most reliable information, consider looking at on-chain metrics, development activity, and adoption rates alongside price action.",
                "That's an interesting question about crypto. While market conditions fluctuate, focusing on fundamentals and having a clear investment strategy tends to yield better long-term results.",
                "Thanks for reaching out. The crypto ecosystem continues to evolve rapidly, making it important to stay informed from multiple sources and maintain a balanced perspective on market developments."
            ]
        
        response = random.choice(templates)
        logger.info(f"Generated template response to query: {query[:50]}...")
        return response