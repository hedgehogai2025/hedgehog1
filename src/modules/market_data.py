import os
import requests
import pandas as pd
import logging
import json
from datetime import datetime, timedelta
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/market_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

class MarketData:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.cache_dir = "data/market_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_top_coins(self, limit=100):
        """Get data for top cryptocurrencies by market cap."""
        endpoint = f"/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": False
        }
        
        try:
            response = self._make_request(endpoint, params)
            
            if response:
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(response)
                logger.info(f"Retrieved data for top {len(df)} coins by market cap")
                return df
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error fetching top coins: {str(e)}")
            return pd.DataFrame()
    
    def get_coin_price_history(self, coin_id, days=30):
        """Get historical price data for a specific coin."""
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }
        
        cache_file = f"{self.cache_dir}/{coin_id}_price_history_{days}.json"
        # Check if we have fresh cached data (less than 6 hours old)
        if os.path.exists(cache_file) and self._is_cache_fresh(cache_file, hours=6):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Retrieved cached price history for {coin_id}")
                    return data
            except Exception:
                # If there's an error with the cache, proceed to fetch new data
                pass
        
        try:
            response = self._make_request(endpoint, params)
            
            if response:
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(response, f)
                
                logger.info(f"Retrieved price history for {coin_id} over {days} days")
                return response
            return None
        
        except Exception as e:
            logger.error(f"Error fetching price history for {coin_id}: {str(e)}")
            return None
    
    def get_trending_coins(self):
        """Get trending search coins on CoinGecko."""
        endpoint = "/search/trending"
        
        try:
            response = self._make_request(endpoint)
            
            if response and 'coins' in response:
                trending = [item['item'] for item in response['coins']]
                logger.info(f"Retrieved {len(trending)} trending coins")
                return trending
            return []
        
        except Exception as e:
            logger.error(f"Error fetching trending coins: {str(e)}")
            return []
    
    def calculate_market_indicators(self, top_coins_df):
        """Calculate various market indicators from top coins data."""
        if top_coins_df.empty:
            return {}
            
        try:
            # Calculate market dominance
            total_market_cap = top_coins_df['market_cap'].sum()
            bitcoin_dominance = (top_coins_df.loc[top_coins_df['id'] == 'bitcoin', 'market_cap'].sum() / total_market_cap) * 100
            ethereum_dominance = (top_coins_df.loc[top_coins_df['id'] == 'ethereum', 'market_cap'].sum() / total_market_cap) * 100
            
            # Calculate 24h market movement
            market_cap_change_percentage = top_coins_df['market_cap_change_percentage_24h'].mean()
            price_change_percentage = top_coins_df['price_change_percentage_24h'].mean()
            
            # Calculate volatility (std dev of price changes)
            volatility = top_coins_df['price_change_percentage_24h'].std()
            
            # Winners and losers
            top_gainers = top_coins_df.nlargest(5, 'price_change_percentage_24h')[['name', 'symbol', 'price_change_percentage_24h']]
            top_losers = top_coins_df.nsmallest(5, 'price_change_percentage_24h')[['name', 'symbol', 'price_change_percentage_24h']]
            
            indicators = {
                'total_market_cap': total_market_cap,
                'bitcoin_dominance': bitcoin_dominance,
                'ethereum_dominance': ethereum_dominance,
                'market_cap_change_percentage_24h': market_cap_change_percentage,
                'average_price_change_percentage_24h': price_change_percentage,
                'market_volatility_24h': volatility,
                'top_gainers': top_gainers.to_dict('records'),
                'top_losers': top_losers.to_dict('records')
            }
            
            # Convert numpy types to standard Python types
            indicators = convert_numpy_types(indicators)
            
            logger.info("Calculated market indicators successfully")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating market indicators: {str(e)}")
            return {}
    
    def _make_request(self, endpoint, params=None):
        """Make a request to the CoinGecko API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params)
            
            # Check if we hit rate limits
            if response.status_code == 429:
                logger.warning("Rate limit hit, waiting before retry")
                # Wait 60 seconds before retrying
                import time
                time.sleep(60)
                response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error making API request to {url}: {str(e)}")
            return None
    
    def _is_cache_fresh(self, file_path, hours=6):
        """Check if a cached file is less than specified hours old."""
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return file_time > cutoff_time