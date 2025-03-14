import os
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import tiktoken
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIAnalyzer:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 cache_dir: str = "cache"):
        """Initialize OpenAI analyzer with API key and settings."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not provided or found in environment variables")
            raise ValueError("OpenAI API key is required")
        
        self.model = model
        self.cache_dir = cache_dir
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize token counter for this session
        self.session_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        # Initialize encoding for token counting
        self.encoding = tiktoken.encoding_for_model(model)
        
        logger.info(f"OpenAI Analyzer initialized with model: {model}")
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate a cache key from messages."""
        # Convert messages to string and hash
        message_str = json.dumps(messages, sort_keys=True)
        import hashlib
        return hashlib.md5(message_str.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if a response is cached for the given key."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            # Check if cache is less than 24 hours old
            file_time = os.path.getmtime(cache_file)
            current_time = time.time()
            if current_time - file_time < 86400:  # 24 hours in seconds
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error reading cache file: {str(e)}")
        return None
    
    def _save_to_cache(self, cache_key: str, response: Dict[str, Any]):
        """Save a response to the cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    def query_openai(self, 
                    messages: List[Dict[str, str]], 
                    temperature: float = 0.7,
                    max_tokens: int = 1000,
                    use_cache: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Send a query to OpenAI API and get a response.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum tokens in the response
            use_cache: Whether to use cached responses
            
        Returns:
            Tuple of (response_text, usage_stats)
        """
        if not self.api_key:
            return "OpenAI API key not configured", {"error": "No API key"}
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(messages)
            cached_response = self._check_cache(cache_key)
            if cached_response:
                logger.info("Using cached response")
                return cached_response["content"], cached_response["usage"]
        
        # Count prompt tokens for tracking
        prompt_tokens = 0
        for message in messages:
            prompt_tokens += self._count_tokens(message["content"])
        
        try:
            # Make the API request with new client format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Update token counts
            self.session_tokens["prompt_tokens"] += usage.get("prompt_tokens", prompt_tokens)
            self.session_tokens["completion_tokens"] += usage.get("completion_tokens", self._count_tokens(response_text))
            self.session_tokens["total_tokens"] += usage.get("total_tokens", 
                                                          usage.get("prompt_tokens", prompt_tokens) + 
                                                          usage.get("completion_tokens", self._count_tokens(response_text)))
            
            # Cache the response if enabled
            if use_cache:
                cache_data = {
                    "content": response_text,
                    "usage": usage,
                    "timestamp": datetime.now().isoformat()
                }
                self._save_to_cache(self._get_cache_key(messages), cache_data)
            
            logger.info(f"OpenAI API call successful, used {usage.get('total_tokens', 'unknown')} tokens")
            return response_text, usage
        except Exception as e:
            error_msg = f"Error querying OpenAI API: {str(e)}"
            logger.error(error_msg)
            return f"Error: {str(e)}", {"error": str(e)}
    
    def get_session_token_usage(self) -> Dict[str, int]:
        """Get the token usage for the current session."""
        return self.session_tokens
    
    def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze cryptocurrency market data and provide insights.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary with analysis results and insights
        """
        # Extract relevant data for analysis
        top_coins = market_data.get('top_coins', [])[:10]  # Top 10 coins
        market_cap = market_data.get('total_market_cap', 0)
        market_cap_change = market_data.get('market_cap_change_percentage_24h', 0)
        trending_coins = market_data.get('trending_coins', [])
        
        # Format data for OpenAI prompt
        formatted_data = {
            "market_overview": {
                "total_market_cap": f"${market_cap/1e9:.2f}B",
                "market_cap_change_24h": f"{market_cap_change:.2f}%",
                "trend": "bullish" if market_cap_change > 0 else "bearish"
            },
            "top_coins": [
                {
                    "symbol": coin.get('symbol', '').upper(),
                    "price": coin.get('current_price', 0),
                    "change_24h": coin.get('price_change_percentage_24h', 0),
                    "market_cap": coin.get('market_cap', 0)
                }
                for coin in top_coins
            ],
            "trending_coins": [
                {
                    "name": coin.get('name', ''),
                    "symbol": coin.get('symbol', '').upper()
                }
                for coin in trending_coins
            ]
        }
        
        # Create prompt for OpenAI
        prompt = f"""
        You are a professional cryptocurrency market analyst. Analyze the following market data and provide insights:
        
        {json.dumps(formatted_data, indent=2)}
        
        Provide the following in your analysis:
        1. Overall market sentiment and direction
        2. Top performing coins and potential reasons
        3. Underperforming coins and potential concerns
        4. Key market trends to watch
        5. Short-term price movement predictions (24-48 hours)
        
        Format your analysis in a concise, professional manner suitable for investors.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert cryptocurrency market analyst with years of experience."},
            {"role": "user", "content": prompt}
        ]
        
        # Query OpenAI
        analysis_text, usage = self.query_openai(messages, temperature=0.4)
        
        # Process the analysis text into structured data
        analysis_sections = analysis_text.split('\n\n')
        
        # Extract predictions if possible
        predictions = {}
        for coin in top_coins[:5]:  # Focus on top 5 coins
            symbol = coin.get('symbol', '').upper()
            # Look for mentions of the coin in the analysis
            for section in analysis_sections:
                if symbol in section:
                    if "bullish" in section.lower() or "increase" in section.lower() or "rise" in section.lower():
                        predictions[symbol] = "bullish"
                    elif "bearish" in section.lower() or "decrease" in section.lower() or "fall" in section.lower():
                        predictions[symbol] = "bearish"
                    else:
                        predictions[symbol] = "neutral"
                    break
            
            # Default to neutral if not found
            if symbol not in predictions:
                predictions[symbol] = "neutral"
        
        # Structure the results
        result = {
            "market_sentiment": "bullish" if market_cap_change > 0 else "bearish",
            "analysis_text": analysis_text,
            "predictions": predictions,
            "token_usage": usage
        }
        
        return result
    
    def predict_trends(self, 
                      historical_data: Dict[str, Any], 
                      social_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict cryptocurrency market trends based on historical and social data.
        
        Args:
            historical_data: Dictionary containing historical market data
            social_data: Dictionary containing social media sentiment and trends
            
        Returns:
            Dictionary with trend predictions and insights
        """
        # Extract relevant historical data
        historical_prices = {}
        if 'price_history' in historical_data:
            for symbol, prices in historical_data['price_history'].items():
                # Get last 7 days if available
                historical_prices[symbol] = prices[-7:] if len(prices) >= 7 else prices
        
        # Extract sentiment data
        sentiment_data = {}
        if 'sentiment' in social_data:
            sentiment_data = social_data['sentiment']
        
        # Extract top topics/trends
        topics = []
        if 'topics' in social_data:
            topics = list(social_data['topics'].keys())[:5]  # Top 5 topics
        
        # Format data for OpenAI prompt
        formatted_data = {
            "historical_prices": historical_prices,
            "sentiment": sentiment_data,
            "trending_topics": topics
        }
        
        # Create prompt for OpenAI
        prompt = f"""
        You are a cryptocurrency trend analyst with expertise in technical analysis and social sentiment. 
        Analyze the following data and predict market trends for the next 7 days:
        
        {json.dumps(formatted_data, indent=2)}
        
        Provide the following in your analysis:
        1. Overall market trend prediction
        2. Specific predictions for major cryptocurrencies
        3. Sentiment-based insights
        4. Trending topics and their potential impact
        5. Key levels to watch (support/resistance)
        
        Format your analysis in a professional and actionable manner.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert cryptocurrency analyst with advanced technical and fundamental analysis skills."},
            {"role": "user", "content": prompt}
        ]
        
        # Query OpenAI
        prediction_text, usage = self.query_openai(messages, temperature=0.3)
        
        # Extract key predictions
        prediction_lines = prediction_text.split('\n')
        key_predictions = []
        
        for line in prediction_lines:
            if any(keyword in line.lower() for keyword in ["predict", "expect", "likely", "probable", "may", "could"]):
                # Clean up the line
                clean_line = line.strip()
                if clean_line and len(clean_line) > 10:  # Minimum meaningful length
                    key_predictions.append(clean_line)
        
        # Structure the results
        result = {
            "full_analysis": prediction_text,
            "key_predictions": key_predictions[:5],  # Top 5 key predictions
            "token_usage": usage
        }
        
        return result
    
    def generate_market_update(self, market_data: Dict[str, Any]) -> str:
        """
        Generate a concise market update suitable for Twitter.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Formatted market update text
        """
        # Extract key data
        top_coins = market_data.get('top_coins', [])[:5]  # Top 5 coins
        market_cap = market_data.get('total_market_cap', 0)
        market_cap_change = market_data.get('market_cap_change_percentage_24h', 0)
        
        # Format data for prompt
        coin_data = []
        for coin in top_coins:
            coin_data.append({
                "symbol": coin.get('symbol', '').upper(),
                "price": coin.get('current_price', 0),
                "change_24h": coin.get('price_change_percentage_24h', 0)
            })
        
        formatted_data = {
            "market_cap": f"${market_cap/1e9:.2f}B",
            "market_cap_change": f"{market_cap_change:.2f}%",
            "top_coins": coin_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        # Create prompt for OpenAI
        prompt = f"""
        Generate a concise cryptocurrency market update tweet with the following data:
        
        {json.dumps(formatted_data, indent=2)}
        
        Requirements:
        1. Must be under 280 characters
        2. Include price and 24h change for top coins
        3. Use emojis appropriately (🚀 for big gains, 📉 for losses, etc.)
        4. Include relevant hashtags
        5. Sound professional but engaging
        """
        
        messages = [
            {"role": "system", "content": "You are a cryptocurrency market analyst creating engaging Twitter updates."},
            {"role": "user", "content": prompt}
        ]
        
        # Query OpenAI
        tweet_text, _ = self.query_openai(messages, temperature=0.7, max_tokens=280)
        
        # Ensure the tweet is under 280 characters
        if len(tweet_text) > 280:
            tweet_text = tweet_text[:277] + "..."
        
        return tweet_text
    
    def generate_response_to_mention(self, 
                                   mention_text: str, 
                                   user: str, 
                                   market_data: Dict[str, Any]) -> str:
        """
        Generate a personalized response to a Twitter mention.
        
        Args:
            mention_text: The text of the mention
            user: Username of the person who mentioned the bot
            market_data: Current market data for context
            
        Returns:
            Response text suitable for Twitter
        """
        # Extract potential cryptocurrency symbols from the mention
        import re
        # Look for symbols like $BTC, $ETH, etc.
        symbols = re.findall(r'\$([A-Za-z0-9]{2,10})', mention_text)
        # Also look for common names
        common_names = {
            'bitcoin': 'BTC', 
            'ethereum': 'ETH', 
            'solana': 'SOL',
            'cardano': 'ADA',
            'binance': 'BNB',
            'ripple': 'XRP',
            'dogecoin': 'DOGE'
        }
        
        for name, symbol in common_names.items():
            if name.lower() in mention_text.lower() and symbol not in symbols:
                symbols.append(symbol)
        
        # Get data for mentioned coins
        coin_data = []
        for coin in market_data.get('top_coins', []):
            if coin.get('symbol', '').upper() in symbols:
                coin_data.append({
                    "symbol": coin.get('symbol', '').upper(),
                    "price": coin.get('current_price', 0),
                    "change_24h": coin.get('price_change_percentage_24h', 0),
                    "market_cap": coin.get('market_cap', 0),
                    "volume": coin.get('total_volume', 0)
                })
        
        # Determine if this is a question
        is_question = '?' in mention_text or any(q in mention_text.lower() for q in [
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 
            'can you', 'could you', 'will', 'should', 'tell me'
        ])
        
        # Create context for OpenAI
        context = {
            "mention": mention_text,
            "user": user,
            "is_question": is_question,
            "mentioned_coins": coin_data,
            "market_data": {
                "total_market_cap": market_data.get('total_market_cap', 0) / 1e9,  # In billions
                "market_sentiment": "bullish" if market_data.get('market_cap_change_percentage_24h', 0) > 0 else "bearish"
            }
        }
        
        # Create prompt for OpenAI
        prompt = f"""
        You are a cryptocurrency analysis bot responding to a Twitter mention. Generate a helpful, personalized response.
        
        Mention context:
        {json.dumps(context, indent=2)}
        
        Guidelines for your response:
        1. Be concise (under 280 characters)
        2. If specific coins were mentioned, include their current data
        3. If it's a question, provide a helpful answer
        4. If no specific coins or questions, give general market insight
        5. Use a friendly but professional tone
        6. Include relevant emojis where appropriate
        7. Sign the tweet with #AI at the end
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful cryptocurrency analysis assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Query OpenAI
        response_text, _ = self.query_openai(messages, temperature=0.7, max_tokens=280)
        
        # Ensure the response is under 280 characters
        if len(response_text) > 280:
            response_text = response_text[:277] + "..."
        
        return response_text
    
    def analyze_social_sentiment(self, social_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform advanced sentiment analysis on social media data.
        
        Args:
            social_data: Dictionary containing social media posts and data
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Extract posts for analysis
        posts = []
        
        # Reddit posts
        if 'reddit_posts' in social_data:
            for post in social_data['reddit_posts'][:20]:  # Limit to 20 posts
                posts.append({
                    "source": "Reddit",
                    "text": f"{post.get('title', '')} {post.get('body', '')}",
                    "score": post.get('score', 0)
                })
        
        # Twitter posts/tweets
        if 'influencer_tweets' in social_data:
            for user, tweets in social_data['influencer_tweets'].items():
                for tweet in tweets[:3]:  # Top 3 tweets per influencer
                    posts.append({
                        "source": f"Twitter/{user}",
                        "text": tweet.get('text', ''),
                        "likes": tweet.get('likes', 0)
                    })
        
        # Format for OpenAI prompt
        formatted_data = {
            "posts": posts[:30]  # Limit total posts to 30 for token conservation
        }
        
        # Create prompt for OpenAI
        prompt = f"""
        You are a cryptocurrency sentiment analyst. Analyze the following social media posts to determine market sentiment:
        
        {json.dumps(formatted_data, indent=2)}
        
        Provide the following in your analysis:
        1. Overall market sentiment (bullish, neutral, or bearish)
        2. Sentiment breakdown for major cryptocurrencies mentioned
        3. Key concerns or excitement factors in the community
        4. Emerging trends or narratives
        5. Confidence level in your assessment (low, medium, high)
        
        Format your response as structured JSON.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in cryptocurrency market sentiment analysis."},
            {"role": "user", "content": prompt}
        ]
        
        # Query OpenAI
        analysis_text, usage = self.query_openai(messages, temperature=0.3)
        
        # Try to parse as JSON
        try:
            analysis_json = json.loads(analysis_text)
            return analysis_json
        except json.JSONDecodeError:
            # If not valid JSON, return as text
            return {
                "text_analysis": analysis_text,
                "format_error": "Response was not valid JSON",
                "token_usage": usage
            }
    
    def generate_technical_analysis(self, coin_symbol: str, ohlc_data: Any) -> str:
        """
        Generate a technical analysis report for a specific cryptocurrency.
        
        Args:
            coin_symbol: Symbol of the cryptocurrency (e.g., 'BTC')
            ohlc_data: OHLC (Open, High, Low, Close) price data
            
        Returns:
            Technical analysis text
        """
        # Convert OHLC data to a suitable format
        data_points = []
        
        try:
            # Assuming ohlc_data is a pandas DataFrame with columns like 'open', 'high', 'low', 'close'
            # Convert last 10 data points to list
            for _, row in ohlc_data.tail(10).iterrows():
                data_points.append({
                    "date": str(row.name),  # Assuming the index is the timestamp
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']) if 'volume' in row else 0
                })
        except Exception as e:
            logger.error(f"Error processing OHLC data: {str(e)}")
            data_points = [{"error": "Failed to process OHLC data"}]
        
        # Create prompt for OpenAI
        prompt = f"""
        You are a cryptocurrency technical analyst. Analyze the following OHLC data for {coin_symbol} and provide a technical analysis:
        
        {json.dumps(data_points, indent=2)}
        
        Include in your analysis:
        1. Price action and trend direction
        2. Support and resistance levels
        3. Key technical indicators (assume standard indicators like RSI, MACD, etc.)
        4. Potential chart patterns
        5. Short-term price targets
        
        Format your analysis as a concise technical report suitable for traders.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in cryptocurrency technical analysis."},
            {"role": "user", "content": prompt}
        ]
        
        # Query OpenAI
        analysis_text, _ = self.query_openai(messages, temperature=0.4)
        
        return analysis_text
    
    def generate_trading_signal(self, 
                              coin_symbol: str, 
                              market_data: Dict[str, Any],
                              technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading signal recommendation for a specific cryptocurrency.
        
        Args:
            coin_symbol: Symbol of the cryptocurrency (e.g., 'BTC')
            market_data: Current market data
            technical_data: Technical indicators and analysis
            
        Returns:
            Dictionary with trading signal information
        """
        # Find the coin data
        coin_data = None
        for coin in market_data.get('top_coins', []):
            if coin.get('symbol', '').upper() == coin_symbol.upper():
                coin_data = coin
                break
        
        if not coin_data:
            return {
                "symbol": coin_symbol,
                "signal_type": "neutral",
                "strength": 0.5,
                "reason": "Insufficient data available for analysis"
            }
        
        # Extract technical indicators
        technical_indicators = technical_data.get(coin_symbol.upper(), {})
        
        # Format data for OpenAI
        formatted_data = {
            "coin": coin_symbol,
            "current_price": coin_data.get('current_price', 0),
            "market_data": {
                "price_change_24h": coin_data.get('price_change_percentage_24h', 0),
                "price_change_7d": coin_data.get('price_change_percentage_7d', 0),
                "market_cap": coin_data.get('market_cap', 0),
                "volume": coin_data.get('total_volume', 0)
            },
            "technical_indicators": technical_indicators
        }
        
        # Create prompt for OpenAI
        prompt = f"""
        You are a cryptocurrency trading analyst. Based on the following market and technical data, generate a trading signal for {coin_symbol}:
        
        {json.dumps(formatted_data, indent=2)}
        
        Provide your response in the following JSON format:
        {{
            "signal_type": "buy", "sell", or "neutral",
            "strength": [a value from 0.0 to 1.0 indicating confidence],
            "price_target": [price level expected],
            "stop_loss": [suggested stop loss],
            "timeframe": "short_term", "medium_term", or "long_term",
            "reasoning": [brief explanation of the signal]
        }}
        
        Be decisive and clear in your recommendation.
        """
        
        messages = [
            {"role": "system", "content": "You are an experienced cryptocurrency trader and analyst."},
            {"role": "user", "content": prompt}
        ]
        
        # Query OpenAI
        signal_text, _ = self.query_openai(messages, temperature=0.4)
        
        # Try to parse as JSON
        try:
            signal_data = json.loads(signal_text)
            # Add the symbol
            signal_data["symbol"] = coin_symbol.upper()
            return signal_data
        except json.JSONDecodeError:
            # If not valid JSON, extract signal manually
            signal_type = "neutral"
            strength = 0.5
            
            if "buy" in signal_text.lower() or "bullish" in signal_text.lower():
                signal_type = "buy"
                strength = 0.7
            elif "sell" in signal_text.lower() or "bearish" in signal_text.lower():
                signal_type = "sell"
                strength = 0.7
            
            return {
                "symbol": coin_symbol.upper(),
                "signal_type": signal_type,
                "strength": strength,
                "text_response": signal_text,
                "format_error": "Response was not valid JSON"
            }