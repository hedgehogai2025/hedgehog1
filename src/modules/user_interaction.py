import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class UserInteractionHandler:
    def __init__(self, twitter_client=None, data_dir: str = "data"):
        """Initialize user interaction handler with Twitter client and data directory."""
        self.twitter_client = twitter_client
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Create interaction history file if it doesn't exist
        self.interactions_file = os.path.join(data_dir, "interaction_history.json")
        if not os.path.exists(self.interactions_file):
            with open(self.interactions_file, "w") as f:
                json.dump({"last_mention_id": None, "interactions": []}, f)
    
    def _load_interaction_history(self) -> Dict[str, Any]:
        """Load interaction history from file."""
        try:
            with open(self.interactions_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading interaction history: {str(e)}")
            return {"last_mention_id": None, "interactions": []}
    
    def _save_interaction_history(self, history: Dict[str, Any]):
        """Save interaction history to file."""
        try:
            with open(self.interactions_file, "w") as f:
                json.dump(history, f, default=str)
        except Exception as e:
            logger.error(f"Error saving interaction history: {str(e)}")
    
    def check_mentions(self) -> List[Dict[str, Any]]:
        """Check for new mentions on Twitter."""
        if not self.twitter_client:
            logger.error("Twitter client not initialized")
            return []
        
        logger.info("Checking for new mentions...")
        
        # Load interaction history
        history = self._load_interaction_history()
        last_mention_id = history.get("last_mention_id")
        
        # Fetch mentions
        mentions = self.twitter_client.get_mentions(since_id=last_mention_id)
        
        if not mentions:
            logger.info("No new mentions found")
            return []
        
        # Update last mention ID
        if mentions:
            history["last_mention_id"] = mentions[0]["id"]
            self._save_interaction_history(history)
        
        logger.info(f"Found {len(mentions)} new mentions")
        return mentions
    
    def process_command(self, text: str) -> Dict[str, Any]:
        """Process command from user interaction."""
        text_lower = text.lower()
        
        # Command format: @bot_name command [parameters]
        command_result = {
            "command": "unknown",
            "parameters": {},
            "response": "I didn't understand that command. Try 'help' for a list of commands."
        }
        
        # Help command
        if "help" in text_lower:
            command_result["command"] = "help"
            command_result["response"] = """
🤖 Commands:
- price [coin]: Get current price and 24h change
- analysis [coin]: Get technical analysis
- sentiment [coin]: Get sentiment analysis
- signals: Get top trading signals
- market: Get market overview
- help: Show this help message

Example: @hedgehogai2025 price bitcoin
            """.strip()
        
        # Price command
        elif "price" in text_lower:
            command_result["command"] = "price"
            # Extract coin parameter
            words = text_lower.split()
            idx = words.index("price") if "price" in words else -1
            
            if idx >= 0 and idx + 1 < len(words):
                coin = words[idx + 1]
                command_result["parameters"]["coin"] = coin
                command_result["response"] = f"Getting price information for {coin.upper()}..."
            else:
                command_result["response"] = "Please specify a coin. Example: price bitcoin"
        
        # Analysis command
        elif "analysis" in text_lower:
            command_result["command"] = "analysis"
            # Extract coin parameter
            words = text_lower.split()
            idx = words.index("analysis") if "analysis" in words else -1
            
            if idx >= 0 and idx + 1 < len(words):
                coin = words[idx + 1]
                command_result["parameters"]["coin"] = coin
                command_result["response"] = f"Generating technical analysis for {coin.upper()}..."
            else:
                command_result["response"] = "Please specify a coin. Example: analysis ethereum"
        
        # Sentiment command
        elif "sentiment" in text_lower:
            command_result["command"] = "sentiment"
            # Extract coin parameter
            words = text_lower.split()
            idx = words.index("sentiment") if "sentiment" in words else -1
            
            if idx >= 0 and idx + 1 < len(words):
                coin = words[idx + 1]
                command_result["parameters"]["coin"] = coin
                command_result["response"] = f"Analyzing sentiment for {coin.upper()}..."
            else:
                command_result["response"] = "Please specify a coin. Example: sentiment bitcoin"
        
        # Signals command
        elif "signals" in text_lower or "trading" in text_lower:
            command_result["command"] = "signals"
            command_result["response"] = "Generating current trading signals..."
        
        # Market command
        elif "market" in text_lower:
            command_result["command"] = "market"
            command_result["response"] = "Generating market overview..."
        
        return command_result
    
    def handle_mention(self, mention: Dict[str, Any], market_data: Dict[str, Any], social_data: Dict[str, Any]) -> bool:
        """Handle a mention and respond to the user."""
        if not self.twitter_client:
            logger.error("Twitter client not initialized")
            return False
        
        try:
            mention_id = mention["id"]
            mention_text = mention["text"]
            username = mention["username"]
            
            logger.info(f"Processing mention from @{username}: {mention_text}")
            
            # Process command
            command_result = self.process_command(mention_text)
            command = command_result["command"]
            parameters = command_result["parameters"]
            
            # Initial response
            initial_response = command_result["response"]
            
            # Send initial response
            self.twitter_client.reply_to_tweet(mention_id, initial_response)
            
            # Handle specific commands
            if command == "price":
                coin = parameters.get("coin", "").lower()
                
                # Find coin in market data
                coin_data = None
                for c in market_data.get("top_coins", []):
                    if c["symbol"].lower() == coin or c["id"].lower() == coin:
                        coin_data = c
                        break
                
                if coin_data:
                    price = coin_data["current_price"]
                    change_24h = coin_data["price_change_percentage_24h"]
                    market_cap = coin_data["market_cap"]
                    volume = coin_data["total_volume"]
                    
                    response = f"""
📊 {coin_data['name']} (${coin_data['symbol'].upper()})
💰 Price: ${price:,.2f}
📈 24h Change: {change_24h:.2f}% {"🟢" if change_24h >= 0 else "🔴"}
💼 Market Cap: ${market_cap:,.0f}
🔄 24h Volume: ${volume:,.0f}
                    """.strip()
                    
                    self.twitter_client.reply_to_tweet(mention_id, response)
                else:
                    self.twitter_client.reply_to_tweet(mention_id, f"Sorry, I couldn't find data for {coin.upper()}.")
            
            elif command == "analysis":
                coin = parameters.get("coin", "").lower()
                
                # Find coin in market data
                coin_data = None
                for c in market_data.get("top_coins", []):
                    if c["symbol"].lower() == coin or c["id"].lower() == coin:
                        coin_data = c
                        break
                
                if coin_data:
                    symbol = coin_data["symbol"].upper()
                    tech_data = market_data.get("technical_data", {}).get(symbol, {})
                    
                    if tech_data:
                        trend = tech_data.get("trend", "Unknown")
                        rsi = tech_data.get("rsi", 0)
                        rsi_signal = tech_data.get("rsi_signal", "Unknown")
                        macd_signal = tech_data.get("macd_signal", "Unknown")
                        summary = tech_data.get("summary", "Mixed signals")
                        
                        response = f"""
📈 Technical Analysis: {coin_data['name']} (${symbol})

Trend: {trend}
RSI: {rsi:.2f} ({rsi_signal})
MACD: {macd_signal}

Summary: {summary}

Note: This is not financial advice.
                        """.strip()
                        
                        self.twitter_client.reply_to_tweet(mention_id, response)
                    else:
                        self.twitter_client.reply_to_tweet(mention_id, f"Sorry, I don't have technical data for {symbol}.")
                else:
                    self.twitter_client.reply_to_tweet(mention_id, f"Sorry, I couldn't find {coin.upper()} in my data.")
            
            elif command == "sentiment":
                coin = parameters.get("coin", "").lower()
                
                # Map symbol to entity name if needed
                entity_map = {
                    "btc": "bitcoin",
                    "eth": "ethereum",
                    "sol": "solana",
                    "doge": "dogecoin",
                    "ada": "cardano",
                    "xrp": "ripple"
                }
                
                entity = entity_map.get(coin, coin)
                
                # Find sentiment in social data
                sentiment_data = social_data.get("sentiment", {}).get(entity)
                
                if sentiment_data:
                    overall = sentiment_data.get("overall", "Neutral")
                    score = sentiment_data.get("overall_score", 0) * 100
                    positive = sentiment_data.get("positive", 0)
                    negative = sentiment_data.get("negative", 0)
                    neutral = sentiment_data.get("neutral", 0)
                    total = sentiment_data.get("total", 0)
                    
                    response = f"""
🔍 Sentiment Analysis: {entity.title()}

Overall: {overall} ({score:.2f}%)
Mentions: {total}
Positive: {positive} ({positive/total*100:.1f}%)
Neutral: {neutral} ({neutral/total*100:.1f}%)
Negative: {negative} ({negative/total*100:.1f}%)

Based on Reddit posts, tweets, and news articles.
                    """.strip()
                    
                    self.twitter_client.reply_to_tweet(mention_id, response)
                else:
                    self.twitter_client.reply_to_tweet(mention_id, f"Sorry, I don't have enough sentiment data for {entity.title()}.")
            
            elif command == "signals":
                signals = market_data.get("trading_signals", [])
                
                if signals:
                    top_signals = signals[:5]  # Get top 5 signals
                    
                    response = "📊 Current Trading Signals:\n\n"
                    
                    for signal in top_signals:
                        symbol = signal["symbol"]
                        signal_type = signal["signal_type"].upper()
                        strength = signal["strength"]
                        emoji = "🟢" if signal_type == "BUY" else "🔴"
                        
                        response += f"{emoji} {signal_type} {symbol}: Strength {strength:.2f}/1.0\n"
                    
                    response += "\nDISCLAIMER: Not financial advice."
                    
                    self.twitter_client.reply_to_tweet(mention_id, response)
                else:
                    self.twitter_client.reply_to_tweet(mention_id, "Sorry, I don't have any trading signals at the moment.")
            
            elif command == "market":
                market_cap = market_data.get("total_market_cap", 0) / 1e9  # Convert to billions
                market_cap_change = market_data.get("market_cap_change_percentage_24h", 0)
                volume = market_data.get("total_volume", 0) / 1e9  # Convert to billions
                
                # Get top performers
                top_coins = sorted(
                    market_data.get("top_coins", [])[:20],
                    key=lambda x: x.get("price_change_percentage_24h", 0),
                    reverse=True
                )[:3]
                
                # Get worst performers
                worst_coins = sorted(
                    market_data.get("top_coins", [])[:20],
                    key=lambda x: x.get("price_change_percentage_24h", 0)
                )[:3]
                
                response = f"""
📊 Crypto Market Overview:

Market Cap: ${market_cap:.2f}B ({market_cap_change:.2f}% {"🟢" if market_cap_change >= 0 else "🔴"})
24h Volume: ${volume:.2f}B

Top Performers:
"""
                
                for coin in top_coins:
                    symbol = coin["symbol"].upper()
                    change = coin["price_change_percentage_24h"]
                    response += f"${symbol}: +{change:.2f}%\n"
                
                response += "\nWorst Performers:\n"
                
                for coin in worst_coins:
                    symbol = coin["symbol"].upper()
                    change = coin["price_change_percentage_24h"]
                    response += f"${symbol}: {change:.2f}%\n"
                
                self.twitter_client.reply_to_tweet(mention_id, response)
            
            # Record interaction in history
            history = self._load_interaction_history()
            history["interactions"].append({
                "mention_id": mention_id,
                "username": username,
                "text": mention_text,
                "command": command,
                "timestamp": datetime.now().isoformat()
            })
            self._save_interaction_history(history)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling mention: {str(e)}")
            return False
    
    def process_interactions(self, market_data: Dict[str, Any], social_data: Dict[str, Any]) -> int:
        """Process all new user interactions."""
        # Check for new mentions
        mentions = self.check_mentions()
        
        processed_count = 0
        
        # Handle each mention
        for mention in mentions:
            success = self.handle_mention(mention, market_data, social_data)
            if success:
                processed_count += 1
        
        return processed_count