import logging
import json
import time
import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketDataCollector:
    def __init__(self, 
                 coingecko_api_key: Optional[str] = None,
                 cryptocompare_api_key: Optional[str] = None,
                 data_dir: str = "data"):
        """Initialize market data collector with API keys and data directory."""
        self.coingecko_api_key = coingecko_api_key
        self.cryptocompare_api_key = cryptocompare_api_key
        self.data_dir = data_dir
        
        # Base URLs for APIs
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.coingecko_pro_url = "https://pro-api.coingecko.com/api/v3"
        self.cryptocompare_base_url = "https://min-api.cryptocompare.com/data"
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Headers for API requests
        self.headers = {"Accept": "application/json"}
        if coingecko_api_key:
            self.headers["x-cg-pro-api-key"] = coingecko_api_key
    
    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make a request to an API with rate limiting and error handling."""
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            # Check if we're being rate limited
            if response.status_code == 429:
                logger.warning(f"Rate limited by API. Sleeping for 60 seconds.")
                time.sleep(60)
                return self._make_request(url, params)  # Try again after waiting
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
    
    def get_top_coins(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch data for top coins by market cap."""
        try:
            # Determine which API to use based on API key
            if self.coingecko_api_key:
                url = f"{self.coingecko_pro_url}/coins/markets"
            else:
                url = f"{self.coingecko_base_url}/coins/markets"
            
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h,7d,30d"
            }
            
            response = self._make_request(url, params)
            
            if not response:
                logger.error("Failed to fetch top coins data")
                return []
            
            logger.info(f"Retrieved data for top {len(response)} coins by market cap")
            return response
        except Exception as e:
            logger.error(f"Error fetching top coins: {str(e)}")
            return []
    
    def get_trending_coins(self) -> List[Dict[str, Any]]:
        """Fetch trending coins from CoinGecko."""
        try:
            # Determine which API to use based on API key
            if self.coingecko_api_key:
                url = f"{self.coingecko_pro_url}/search/trending"
            else:
                url = f"{self.coingecko_base_url}/search/trending"
            
            response = self._make_request(url)
            
            if not response or "coins" not in response:
                logger.error("Failed to fetch trending coins data")
                return []
            
            trending_coins = []
            for item in response["coins"]:
                coin = item["item"]
                trending_coins.append({
                    "id": coin["id"],
                    "name": coin["name"],
                    "symbol": coin["symbol"],
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "thumb": coin.get("thumb"),
                    "score": coin.get("score")
                })
            
            logger.info(f"Retrieved {len(trending_coins)} trending coins")
            return trending_coins
        except Exception as e:
            logger.error(f"Error fetching trending coins: {str(e)}")
            return []
    
    def get_global_market_data(self) -> Dict[str, Any]:
        """Fetch global cryptocurrency market data."""
        try:
            # Determine which API to use based on API key
            if self.coingecko_api_key:
                url = f"{self.coingecko_pro_url}/global"
            else:
                url = f"{self.coingecko_base_url}/global"
            
            response = self._make_request(url)
            
            if not response or "data" not in response:
                logger.error("Failed to fetch global market data")
                return {}
            
            return response["data"]
        except Exception as e:
            logger.error(f"Error fetching global market data: {str(e)}")
            return {}
    
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """Fetch the Fear & Greed Index from Alternative.me API."""
        try:
            url = "https://api.alternative.me/fng/"
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data or not data["data"]:
                logger.error("Failed to fetch Fear & Greed Index")
                return {}
            
            return data["data"][0]
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {str(e)}")
            return {}
    
    def get_ohlc_data(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """Fetch OHLC data for a cryptocurrency."""
        try:
            # Determine which API to use based on API key
            if self.coingecko_api_key:
                url = f"{self.coingecko_pro_url}/coins/{coin_id}/ohlc"
            else:
                url = f"{self.coingecko_base_url}/coins/{coin_id}/ohlc"
            
            params = {
                "vs_currency": "usd",
                "days": days
            }
            
            response = self._make_request(url, params)
            
            if not response:
                logger.error(f"Failed to fetch OHLC data for {coin_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(response, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLC data for {coin_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, ohlc_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various technical indicators from OHLC data."""
        try:
            if ohlc_data.empty:
                logger.error("Cannot calculate indicators from empty OHLC data")
                return {}
            
            indicators = {}
            
            # Simple Moving Averages
            indicators["sma_20"] = ohlc_data["close"].rolling(window=20).mean().iloc[-1]
            indicators["sma_50"] = ohlc_data["close"].rolling(window=50).mean().iloc[-1]
            indicators["sma_200"] = ohlc_data["close"].rolling(window=200).mean().iloc[-1]
            
            # Exponential Moving Averages
            indicators["ema_12"] = ohlc_data["close"].ewm(span=12, adjust=False).mean().iloc[-1]
            indicators["ema_26"] = ohlc_data["close"].ewm(span=26, adjust=False).mean().iloc[-1]
            
            # MACD
            indicators["macd"] = indicators["ema_12"] - indicators["ema_26"]
            indicators["macd_signal"] = pd.Series(indicators["ema_12"] - indicators["ema_26"]).ewm(span=9, adjust=False).mean().iloc[-1]
            indicators["macd_hist"] = indicators["macd"] - indicators["macd_signal"]
            
            # RSI (14-period)
            delta = ohlc_data["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            indicators["rsi"] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Bollinger Bands (20-period, 2 standard deviations)
            sma20 = ohlc_data["close"].rolling(window=20).mean()
            std20 = ohlc_data["close"].rolling(window=20).std()
            indicators["bb_upper"] = (sma20 + 2 * std20).iloc[-1]
            indicators["bb_middle"] = sma20.iloc[-1]
            indicators["bb_lower"] = (sma20 - 2 * std20).iloc[-1]
            
            # Current price
            indicators["current_price"] = ohlc_data["close"].iloc[-1]
            
            # Price relative to indicators
            indicators["price_rel_sma20"] = indicators["current_price"] / indicators["sma_20"] - 1
            indicators["price_rel_sma50"] = indicators["current_price"] / indicators["sma_50"] - 1
            indicators["price_rel_sma200"] = indicators["current_price"] / indicators["sma_200"] - 1
            
            # Generate basic analysis
            analysis = self._generate_analysis(indicators)
            indicators.update(analysis)
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    def _generate_analysis(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic analysis from technical indicators."""
        analysis = {}
        
                    # Trend analysis
        if indicators["price_rel_sma20"] > 0 and indicators["price_rel_sma50"] > 0:
            analysis["trend"] = "Bullish"
        elif indicators["price_rel_sma20"] < 0 and indicators["price_rel_sma50"] < 0:
            analysis["trend"] = "Bearish"
        else:
            analysis["trend"] = "Mixed"
        
        # MACD analysis
        if indicators["macd"] > 0 and indicators["macd_hist"] > 0:
            analysis["macd_signal"] = "Bullish"
        elif indicators["macd"] < 0 and indicators["macd_hist"] < 0:
            analysis["macd_signal"] = "Bearish"
        elif indicators["macd"] > 0 and indicators["macd_hist"] < 0:
            analysis["macd_signal"] = "Weakening"
        else:
            analysis["macd_signal"] = "Strengthening"
        
        # RSI analysis
        if indicators["rsi"] > 70:
            analysis["rsi_signal"] = "Overbought"
        elif indicators["rsi"] < 30:
            analysis["rsi_signal"] = "Oversold"
        else:
            analysis["rsi_signal"] = "Neutral"
        
        # Bollinger Bands analysis
        if indicators["current_price"] > indicators["bb_upper"]:
            analysis["bb_signal"] = "Overbought"
        elif indicators["current_price"] < indicators["bb_lower"]:
            analysis["bb_signal"] = "Oversold"
        else:
            distance_to_upper = indicators["bb_upper"] - indicators["current_price"]
            distance_to_lower = indicators["current_price"] - indicators["bb_lower"]
            band_height = indicators["bb_upper"] - indicators["bb_lower"]
            
            position = distance_to_lower / band_height
            
            if position > 0.7:
                analysis["bb_signal"] = "Upper Band Test"
            elif position < 0.3:
                analysis["bb_signal"] = "Lower Band Test"
            else:
                analysis["bb_signal"] = "Middle Band"
        
        # Overall summary
        bullish_signals = 0
        bearish_signals = 0
        
        if analysis["trend"] == "Bullish":
            bullish_signals += 1
        elif analysis["trend"] == "Bearish":
            bearish_signals += 1
        
        if analysis["macd_signal"] in ["Bullish", "Strengthening"]:
            bullish_signals += 1
        elif analysis["macd_signal"] in ["Bearish", "Weakening"]:
            bearish_signals += 1
        
        if analysis["rsi_signal"] == "Oversold":
            bullish_signals += 1
        elif analysis["rsi_signal"] == "Overbought":
            bearish_signals += 1
        
        if analysis["bb_signal"] in ["Oversold", "Lower Band Test"]:
            bullish_signals += 1
        elif analysis["bb_signal"] in ["Overbought", "Upper Band Test"]:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals + 1:
            analysis["summary"] = "Strong bullish signals"
        elif bullish_signals > bearish_signals:
            analysis["summary"] = "Moderately bullish"
        elif bearish_signals > bullish_signals + 1:
            analysis["summary"] = "Strong bearish signals"
        elif bearish_signals > bullish_signals:
            analysis["summary"] = "Moderately bearish"
        else:
            analysis["summary"] = "Mixed signals"
        
        return analysis
    
    def get_coin_categories(self) -> Dict[str, List[str]]:
        """Fetch coin categories from CoinGecko."""
        try:
            # Determine which API to use based on API key
            if self.coingecko_api_key:
                url = f"{self.coingecko_pro_url}/coins/categories/list"
            else:
                url = f"{self.coingecko_base_url}/coins/categories/list"
            
            response = self._make_request(url)
            
            if not response:
                logger.error("Failed to fetch coin categories")
                return {}
            
            categories = {}
            for category in response:
                categories[category["id"]] = category["name"]
            
            logger.info(f"Retrieved {len(categories)} coin categories")
            return categories
        except Exception as e:
            logger.error(f"Error fetching coin categories: {str(e)}")
            return {}
    
    def get_sector_performance(self) -> Dict[str, float]:
        """Get performance data for different crypto sectors."""
        try:
            # Determine which API to use based on API key
            if self.coingecko_api_key:
                url = f"{self.coingecko_pro_url}/coins/categories"
            else:
                url = f"{self.coingecko_base_url}/coins/categories"
            
            response = self._make_request(url)
            
            if not response:
                logger.error("Failed to fetch sector performance data")
                return {}
            
            sector_performance = {}
            for category in response:
                sector_performance[category["name"]] = category["market_cap_change_24h"]
            
            logger.info(f"Retrieved performance data for {len(sector_performance)} sectors")
            return sector_performance
        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            return {}
    
    def get_historical_market_data(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical market data for a cryptocurrency."""
        try:
            # Determine which API to use based on API key
            if self.coingecko_api_key:
                url = f"{self.coingecko_pro_url}/coins/{coin_id}/market_chart"
            else:
                url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"
            
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }
            
            response = self._make_request(url, params)
            
            if not response or not all(k in response for k in ["prices", "market_caps", "total_volumes"]):
                logger.error(f"Failed to fetch historical data for {coin_id}")
                return pd.DataFrame()
            
            # Process price data
            prices = pd.DataFrame(response["prices"], columns=["timestamp", "price"])
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
            
            # Process market cap data
            market_caps = pd.DataFrame(response["market_caps"], columns=["timestamp", "market_cap"])
            market_caps["timestamp"] = pd.to_datetime(market_caps["timestamp"], unit="ms")
            
            # Process volume data
            volumes = pd.DataFrame(response["total_volumes"], columns=["timestamp", "volume"])
            volumes["timestamp"] = pd.to_datetime(volumes["timestamp"], unit="ms")
            
            # Merge all data
            df = prices.merge(market_caps, on="timestamp").merge(volumes, on="timestamp")
            df = df.set_index("timestamp")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {coin_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_exchange_rates(self) -> Dict[str, float]:
        """Fetch exchange rates for major currencies against USD."""
        try:
            url = f"{self.cryptocompare_base_url}/price"
            
            params = {
                "fsym": "USD",
                "tsyms": "EUR,GBP,JPY,CNY,AUD,CAD"
            }
            
            if self.cryptocompare_api_key:
                params["api_key"] = self.cryptocompare_api_key
            
            response = self._make_request(url, params)
            
            if not response:
                logger.error("Failed to fetch exchange rates")
                return {}
            
            # Invert rates since we want X/USD instead of USD/X
            exchange_rates = {currency: 1/rate for currency, rate in response.items()}
            
            logger.info(f"Retrieved exchange rates for {len(exchange_rates)} currencies")
            return exchange_rates
        except Exception as e:
            logger.error(f"Error fetching exchange rates: {str(e)}")
            return {}
    
    def generate_market_signals(self, 
                              market_data: List[Dict[str, Any]], 
                              technical_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate trading signals based on market and technical data."""
        try:
            signals = []
            
            for coin in market_data:
                symbol = coin["symbol"].upper()
                
                # Skip if we don't have technical data for this coin
                if symbol not in technical_data:
                    continue
                
                tech = technical_data[symbol]
                
                # Determine signal type
                signal_type = "neutral"
                
                # Check trend
                if tech.get("trend") == "Bullish" and tech.get("macd_signal") in ["Bullish", "Strengthening"]:
                    signal_type = "buy"
                elif tech.get("trend") == "Bearish" and tech.get("macd_signal") in ["Bearish", "Weakening"]:
                    signal_type = "sell"
                
                # Check RSI for confirmation or contradiction
                if signal_type == "buy" and tech.get("rsi_signal") == "Overbought":
                    signal_type = "neutral"  # Contradiction
                elif signal_type == "sell" and tech.get("rsi_signal") == "Oversold":
                    signal_type = "neutral"  # Contradiction
                elif signal_type == "neutral" and tech.get("rsi_signal") == "Oversold":
                    signal_type = "buy"  # RSI oversold is a buy signal
                elif signal_type == "neutral" and tech.get("rsi_signal") == "Overbought":
                    signal_type = "sell"  # RSI overbought is a sell signal
                
                # Skip neutral signals
                if signal_type == "neutral":
                    continue
                
                # Calculate signal strength (0.0 to 1.0)
                strength = 0.5  # Default
                
                # Adjust based on technical indicators
                if tech.get("trend") == "Bullish" and signal_type == "buy":
                    strength += 0.1
                elif tech.get("trend") == "Bearish" and signal_type == "sell":
                    strength += 0.1
                
                if tech.get("macd_signal") == "Bullish" and signal_type == "buy":
                    strength += 0.1
                elif tech.get("macd_signal") == "Bearish" and signal_type == "sell":
                    strength += 0.1
                
                if tech.get("rsi_signal") == "Oversold" and signal_type == "buy":
                    strength += 0.2
                elif tech.get("rsi_signal") == "Overbought" and signal_type == "sell":
                    strength += 0.2
                
                if tech.get("bb_signal") == "Oversold" and signal_type == "buy":
                    strength += 0.1
                elif tech.get("bb_signal") == "Overbought" and signal_type == "sell":
                    strength += 0.1
                
                # Cap strength at 1.0
                strength = min(1.0, strength)
                
                # Add signal
                signals.append({
                    "symbol": symbol,
                    "signal_type": signal_type,
                    "strength": strength,
                    "price": coin["current_price"],
                    "timestamp": datetime.now().isoformat()
                })
            
            # Sort by strength (descending)
            signals = sorted(signals, key=lambda x: x["strength"], reverse=True)
            
            logger.info(f"Generated {len(signals)} trading signals")
            return signals
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return []
    
    def collect_market_data(self) -> Dict[str, Any]:
        """Collect comprehensive market data and indicators."""
        try:
            market_data = {}
            
            # Get top coins
            top_coins = self.get_top_coins(limit=100)
            market_data["top_coins"] = top_coins
            
            # Get global market data
            global_data = self.get_global_market_data()
            market_data["total_market_cap"] = global_data.get("total_market_cap", {}).get("usd", 0)
            market_data["total_volume"] = global_data.get("total_24h_volume", {}).get("usd", 0)
            market_data["market_cap_change_percentage_24h"] = global_data.get("market_cap_change_percentage_24h_usd", 0)
            
            # Get trending coins
            trending_coins = self.get_trending_coins()
            market_data["trending_coins"] = trending_coins
            
            # Get sector performance
            sector_performance = self.get_sector_performance()
            market_data["sector_performance"] = sector_performance
            
            # Get fear & greed index
            fear_greed = self.get_fear_greed_index()
            market_data["fear_greed_index"] = fear_greed
            
            # Get technical data for top coins
            technical_data = {}
            for coin in top_coins[:20]:  # Only analyze top 20 for performance
                coin_id = coin["id"]
                symbol = coin["symbol"].upper()
                
                # Get OHLC data
                ohlc_data = self.get_ohlc_data(coin_id)
                if not ohlc_data.empty:
                    # Calculate technical indicators
                    indicators = self.get_technical_indicators(ohlc_data)
                    technical_data[symbol] = indicators
            
            market_data["technical_data"] = technical_data
            
            # Generate trading signals
            signals = self.generate_market_signals(top_coins[:20], technical_data)
            market_data["trading_signals"] = signals
            
            # Save data to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = os.path.join(self.data_dir, f"market_data_{timestamp}.json")
            
            with open(filename, "w") as f:
                json.dump(market_data, f, default=str)
            
            logger.info(f"Market data collected and saved to {filename}")
            
            return market_data
        except Exception as e:
            logger.error(f"Error collecting market data: {str(e)}")
            return {}