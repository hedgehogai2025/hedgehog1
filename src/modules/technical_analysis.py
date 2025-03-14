import pandas as pd
import numpy as np
import os
import logging
import requests
# import talib
import pandas_ta as ta  # 替换talib
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/technical_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    def __init__(self):
        self.cache_dir = "data/price_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define common technical indicator parameters
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_stdev = 2
        
        # CoinGecko API key
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY', 'CG-HKcah6fnuDW3C4cm1S1c952S')
        
        # Symbol mapping for CoinGecko
        self.symbol_to_id = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot',
            'AVAX': 'avalanche-2',
            'MATIC': 'matic-network'
        }
        
    def calculate_indicators(self, symbol, timeframe='daily'):
        """Calculate technical indicators for a given symbol"""
        try:
            # Get price data
            prices = self._get_historical_prices(symbol, timeframe)
            if prices is None or len(prices) < 30:
                logger.error(f"Unable to get sufficient price data for {symbol}")
                return {}
                
            # Calculate basic indicators
            results = {
                'last_price': prices['close'].iloc[-1],
                'volume_24h': prices['volume'].iloc[-1] if 'volume' in prices else 0,
                'change_24h_pct': ((prices['close'].iloc[-1] / prices['close'].iloc[-2]) - 1) * 100,
                'high_24h': prices['high'].iloc[-1],
                'low_24h': prices['low'].iloc[-1],
            }
            
            # Calculate moving averages
            results['ma_50'] = self._calculate_ma(prices['close'], 50)
            results['ma_200'] = self._calculate_ma(prices['close'], 200)
            
            # Calculate RSI
            results['rsi'] = self._calculate_rsi(prices['close'])
            
            # Calculate MACD
            macd_result = self._calculate_macd(prices['close'])
            results.update(macd_result)
            
            # Calculate Bollinger Bands
            bb_result = self._calculate_bollinger_bands(prices['close'])
            results.update(bb_result)
            
            # Calculate support/resistance levels
            sr_levels = self._find_support_resistance(prices)
            results.update(sr_levels)
            
            # Analyze trend
            trend_analysis = self._analyze_trend(prices, results)
            results.update(trend_analysis)
            
            # Calculate key price level
            results['key_level'] = self._calculate_key_level(prices, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
            return {}
            
    def _get_historical_prices(self, symbol, timeframe='daily'):
        """Get historical price data using CoinGecko API"""
        try:
            # First try to read from cache
            cache_file = f"{self.cache_dir}/{symbol}_{timeframe}.csv"
            
            now = datetime.now()
            if os.path.exists(cache_file):
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                # If cache is less than 2 hours old
                if now - file_time < timedelta(hours=2):
                    return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            # Convert symbol to CoinGecko ID
            coin_id = self.symbol_to_id.get(symbol, symbol.lower())
            
            # Get data from CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '200',  # For 200-day moving average
                'interval': 'daily'
            }
            headers = {
                'x-cg-demo-api-key': self.coingecko_api_key
            }
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process data into DataFrame
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                
                # Create DataFrame
                df = pd.DataFrame(prices, columns=['time', 'close'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                
                # Add volume
                if volumes:
                    vol_df = pd.DataFrame(volumes, columns=['time', 'volume'])
                    vol_df['time'] = pd.to_datetime(vol_df['time'], unit='ms')
                    df = df.merge(vol_df, on='time')
                else:
                    df['volume'] = 0
                
                # Estimate OHLC from daily close prices
                df['open'] = df['close'].shift(1)
                df['high'] = df['close'] * 1.01  # Estimate
                df['low'] = df['close'] * 0.99   # Estimate
                
                # Fill first row's open with its close
                if len(df) > 0:
                    df.loc[0, 'open'] = df.loc[0, 'close']
                
                # Set index
                df.set_index('time', inplace=True)
                
                # Save to cache
                df.to_csv(cache_file)
                
                return df
            else:
                logger.error(f"CoinGecko API request failed: {response.status_code}")
                
                # Fall back to CryptoCompare API if CoinGecko fails
                return self._get_historical_prices_from_cryptocompare(symbol, timeframe)
                
        except Exception as e:
            logger.error(f"Error getting price data from CoinGecko for {symbol}: {str(e)}")
            
            # Try CryptoCompare as fallback
            fallback_data = self._get_historical_prices_from_cryptocompare(symbol, timeframe)
            if fallback_data is not None:
                return fallback_data
            
            # If remote APIs fail but we have cache, use it
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
                
            return None
    
    def _get_historical_prices_from_cryptocompare(self, symbol, timeframe='daily'):
        """Fallback method to get data from CryptoCompare"""
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/histo{timeframe}"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': 200  # Get more data for 200-day moving average
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data['Response'] == 'Success':
                    df = pd.DataFrame(data['Data']['Data'])
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Save to cache
                    cache_file = f"{self.cache_dir}/{symbol}_{timeframe}.csv"
                    df.to_csv(cache_file)
                    
                    return df
                else:
                    logger.error(f"CryptoCompare API error: {data['Message']}")
                    return None
            else:
                logger.error(f"CryptoCompare API request failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error in CryptoCompare fallback for {symbol}: {str(e)}")
            return None
            
    def _calculate_ma(self, close_prices, period):
        """Calculate moving average"""
        try:
            # 使用pandas_ta替代talib
            return ta.sma(close_prices, length=period).iloc[-1]
        except:
            # 如果pandas_ta失败，使用pandas原生方法
            return close_prices.rolling(window=period).mean().iloc[-1]
            
    def _calculate_rsi(self, close_prices):
        """Calculate RSI"""
        try:
            # 使用pandas_ta替代talib
            return ta.rsi(close_prices, length=self.rsi_period).iloc[-1]
        except:
            # Manual RSI calculation
            delta = close_prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
            
    def _calculate_macd(self, close_prices):
        """Calculate MACD"""
        try:
            # 使用pandas_ta替代talib
            macd_result = ta.macd(close_prices, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            
            # 提取pandas_ta返回的结果
            macd_line = macd_result[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'].iloc[-1]
            signal_line = macd_result[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'].iloc[-1]
            macd_hist = macd_result[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'].iloc[-1]
            
            return {
                'macd': macd_line,
                'macd_signal': signal_line,
                'macd_hist': macd_hist
            }
        except:
            # Manual MACD calculation
            exp1 = close_prices.ewm(span=self.macd_fast, adjust=False).mean()
            exp2 = close_prices.ewm(span=self.macd_slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
            macd_hist = macd_line - signal_line
            
            return {
                'macd': macd_line.iloc[-1],
                'macd_signal': signal_line.iloc[-1],
                'macd_hist': macd_hist.iloc[-1]
            }
            
    def _calculate_bollinger_bands(self, close_prices):
        """Calculate Bollinger Bands"""
        try:
            # 使用pandas_ta替代talib
            bbands = ta.bbands(close_prices, length=self.bb_period, std=self.bb_stdev)
            
            upper = bbands[f'BBU_{self.bb_period}_{self.bb_stdev}'].iloc[-1]
            middle = bbands[f'BBM_{self.bb_period}_{self.bb_stdev}'].iloc[-1]
            lower = bbands[f'BBL_{self.bb_period}_{self.bb_stdev}'].iloc[-1]
            
            return {
                'bb_upper': upper,
                'bb_middle': middle,
                'bb_lower': lower,
                'bb_width': (upper - lower) / middle
            }
        except:
            # Manual Bollinger Bands calculation
            rolling_mean = close_prices.rolling(window=self.bb_period).mean()
            rolling_std = close_prices.rolling(window=self.bb_period).std()
            
            upper_band = rolling_mean + (rolling_std * self.bb_stdev)
            lower_band = rolling_mean - (rolling_std * self.bb_stdev)
            
            return {
                'bb_upper': upper_band.iloc[-1],
                'bb_middle': rolling_mean.iloc[-1],
                'bb_lower': lower_band.iloc[-1],
                'bb_width': (upper_band.iloc[-1] - lower_band.iloc[-1]) / rolling_mean.iloc[-1]
            }
            
    def _find_support_resistance(self, prices):
        """Find support and resistance levels"""
        try:
            # Get last 30 days of data
            recent_prices = prices.tail(30)
            
            # Find local highs and lows
            price_min = recent_prices['low'].min()
            price_max = recent_prices['high'].max()
            
            # Find recent support levels (lows)
            support_levels = []
            for i in range(5, len(recent_prices) - 5):
                if (recent_prices['low'].iloc[i] <= recent_prices['low'].iloc[i-1] and 
                    recent_prices['low'].iloc[i] <= recent_prices['low'].iloc[i-2] and
                    recent_prices['low'].iloc[i] <= recent_prices['low'].iloc[i+1] and 
                    recent_prices['low'].iloc[i] <= recent_prices['low'].iloc[i+2]):
                    support_levels.append(recent_prices['low'].iloc[i])
            
            # Find recent resistance levels (highs)
            resistance_levels = []
            for i in range(5, len(recent_prices) - 5):
                if (recent_prices['high'].iloc[i] >= recent_prices['high'].iloc[i-1] and 
                    recent_prices['high'].iloc[i] >= recent_prices['high'].iloc[i-2] and
                    recent_prices['high'].iloc[i] >= recent_prices['high'].iloc[i+1] and 
                    recent_prices['high'].iloc[i] >= recent_prices['high'].iloc[i+2]):
                    resistance_levels.append(recent_prices['high'].iloc[i])
            
            # Get recent price
            current_price = prices['close'].iloc[-1]
            
            # Find closest support and resistance
            supports_below = sorted([s for s in support_levels if s < current_price], reverse=True)
            nearest_support = supports_below[0] if supports_below else price_min
            
            resistances_above = sorted([r for r in resistance_levels if r > current_price])
            nearest_resistance = resistances_above[0] if resistances_above else price_max
            
            return {
                'support_level': nearest_support,
                'resistance_level': nearest_resistance
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance levels: {str(e)}")
            return {
                'support_level': prices['low'].min(),
                'resistance_level': prices['high'].max()
            }
            
    def _analyze_trend(self, prices, indicators):
        """Analyze price trend"""
        try:
            # Get indicators
            ma50 = indicators.get('ma_50', 0)
            ma200 = indicators.get('ma_200', 0)
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            bb_width = indicators.get('bb_width', 0)
            
            # Calculate recent price
            current_price = prices['close'].iloc[-1]
            price_20d_ago = prices['close'].iloc[-21] if len(prices) > 21 else prices['close'].iloc[0]
            
            # Determine overall trend direction
            trend_direction = 0  # 0 means neutral
            
            # MA cross check
            if ma50 > ma200:
                trend_direction += 1  # Bullish
            elif ma50 < ma200:
                trend_direction -= 1  # Bearish
                
            # RSI check
            if rsi > 70:
                trend_direction += 0.5  # Overbought
            elif rsi < 30:
                trend_direction -= 0.5  # Oversold
                
            # MACD check
            if macd > macd_signal:
                trend_direction += 0.5
            elif macd < macd_signal:
                trend_direction -= 0.5
                
            # Price trend
            if current_price > price_20d_ago:
                trend_direction += 0.5
            elif current_price < price_20d_ago:
                trend_direction -= 0.5
                
            # Calculate trend strength (0-1 range)
            trend_strength = abs(trend_direction) / 3.5  # Maximum value is 3.5
            if trend_strength > 1:
                trend_strength = 1
                
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {
                'trend_direction': 0,
                'trend_strength': 0
            }
            
    def _calculate_key_level(self, prices, indicators):
        """Calculate key price level"""
        try:
            current_price = prices['close'].iloc[-1]
            
            if 'support_level' in indicators and 'resistance_level' in indicators:
                support = indicators['support_level']
                resistance = indicators['resistance_level']
                
                # If price is close to support, potential buy point
                if current_price < (support * 1.02):
                    return support
                
                # If price is close to resistance, potential sell point
                if current_price > (resistance * 0.98):
                    return resistance
                
                # Otherwise return current price
                return current_price
            
            return current_price
            
        except Exception as e:
            logger.error(f"Error calculating key price level: {str(e)}")
            return prices['close'].iloc[-1]