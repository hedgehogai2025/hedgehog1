import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import os
import requests
from datetime import datetime, timedelta
import mplfinance as mpf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/chart_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChartGenerator:
    def __init__(self):
        # Set chart style
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Set English locale for dates
        plt.rcParams['axes.formatter.use_locale'] = False
        
        # Set Western fonts to avoid Chinese character issues
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        
        self.colors = {
            'primary': '#1DA1F2',  # Twitter blue
            'secondary': '#17BF63',  # Green
            'warning': '#FFAD1F',   # Yellow
            'danger': '#E0245E',    # Red
            'light': '#FFFFFF',     # White
            'dark': '#14171A',      # Dark
            'text': '#657786'       # Text gray
        }
        
    def create_market_overview(self, market_data):
        """Create a market overview chart"""
        try:
            # Extract indicators
            indicators = market_data.get('indicators', {})
            
            # Create chart
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Cryptocurrency Market Overview', fontsize=16)
            
            # Subplot 1: Market cap share
            self._plot_market_dominance(axes[0, 0], indicators)
            
            # Subplot 2: 24h price changes
            self._plot_price_changes(axes[0, 1], market_data)
            
            # Subplot 3: On-chain activity
            self._plot_onchain_activity(axes[1, 0], market_data)
            
            # Subplot 4: Trending tokens
            self._plot_trending_tokens(axes[1, 1], market_data)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            return fig
            
        except Exception as e:
            logger.error(f"Error creating market overview chart: {str(e)}")
            return None
            
    def _plot_market_dominance(self, ax, indicators):
        """Plot market cap share pie chart"""
        try:
            btc_dom = indicators.get('bitcoin_dominance', 40)
            eth_dom = indicators.get('ethereum_dominance', 20)
            others = 100 - btc_dom - eth_dom
            
            labels = ['Bitcoin', 'Ethereum', 'Others']
            sizes = [btc_dom, eth_dom, others]
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['text']]
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                  shadow=False, startangle=90, wedgeprops={'edgecolor': 'white'})
            ax.set_title('Market Share')
            
        except Exception as e:
            logger.error(f"Error plotting market share chart: {str(e)}")
            ax.text(0.5, 0.5, 'Data unavailable', ha='center')
            
    def _plot_price_changes(self, ax, market_data):
        """Plot 24h price changes comparison"""
        try:
            # Get top assets
            indicators = market_data.get('indicators', {})
            
            # Assume extraction from top_coins
            gainers = indicators.get('top_gainers', [])[:5]
            losers = indicators.get('top_losers', [])[:5]
            
            if not gainers or not losers:
                ax.text(0.5, 0.5, 'Price change data unavailable', ha='center')
                return
                
            # Prepare data
            symbols = []
            changes = []
            colors = []
            
            for item in gainers:
                symbols.append(item.get('symbol', '').upper())
                changes.append(item.get('price_change_percentage_24h', 0))
                colors.append(self.colors['secondary'])
                
            for item in losers:
                symbols.append(item.get('symbol', '').upper())
                changes.append(item.get('price_change_percentage_24h', 0))
                colors.append(self.colors['danger'])
                
            # Draw bar chart
            y_pos = np.arange(len(symbols))
            ax.barh(y_pos, changes, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(symbols)
            ax.set_title('24h Price Change (%)')
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            
            # Add percentage labels on each bar
            for i, v in enumerate(changes):
                ax.text(v + 0.5, i, f"{v:.1f}%", va='center')
                
        except Exception as e:
            logger.error(f"Error plotting price changes chart: {str(e)}")
            ax.text(0.5, 0.5, 'Data unavailable', ha='center')
            
    def _plot_onchain_activity(self, ax, market_data):
        """Plot on-chain activity chart"""
        try:
            onchain_metrics = market_data.get('onchain_metrics', {})
            
            if not onchain_metrics:
                ax.text(0.5, 0.5, 'On-chain data unavailable', ha='center')
                return
                
            # Extract BTC and ETH activity
            btc_data = onchain_metrics.get('bitcoin', {})
            eth_data = onchain_metrics.get('ethereum', {})
            
            # Extract active addresses data
            btc_addresses = btc_data.get('active_addresses', 0)
            eth_addresses = eth_data.get('active_addresses', 0)
            
            # Extract transaction volume data
            btc_volume = btc_data.get('transaction_volume', 0) / 1e9  # Convert to billions
            eth_volume = eth_data.get('transaction_volume', 0) / 1e9  # Convert to billions
            
            # Create dual Y-axis chart
            ax2 = ax.twinx()
            
            # Plot active addresses bar chart
            x = [0, 1]
            address_data = [btc_addresses, eth_addresses]
            bars = ax.bar(x, address_data, width=0.4, color=[self.colors['primary'], self.colors['secondary']])
            
            # Plot transaction volume line chart
            volume_data = [btc_volume, eth_volume]
            ax2.plot(x, volume_data, 'o-', color=self.colors['warning'], linewidth=3)
            
            # Set X-axis labels
            ax.set_xticks(x)
            ax.set_xticklabels(['Bitcoin', 'Ethereum'])
            
            # Set Y-axis labels
            ax.set_ylabel('Active Addresses', color=self.colors['primary'])
            ax2.set_ylabel('Transaction Volume (Billion USD)', color=self.colors['warning'])
            
            # Add title
            ax.set_title('On-chain Activity')
            
            # Add data labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f"{height:,.0f}", ha='center', va='bottom')
                        
            for i, val in enumerate(volume_data):
                ax2.text(i, val + 0.1, f"${val:.2f}B", ha='center', va='bottom', color=self.colors['warning'])
                
        except Exception as e:
            logger.error(f"Error plotting on-chain activity chart: {str(e)}")
            ax.text(0.5, 0.5, 'Data unavailable', ha='center')
            
    def _plot_trending_tokens(self, ax, market_data):
        """Plot trending tokens chart"""
        try:
            trending = market_data.get('trending_coins', [])
            
            if not trending or len(trending) < 3:
                ax.text(0.5, 0.5, 'Trending tokens data unavailable', ha='center')
                return
                
            # Prepare data (top 10)
            tokens = []
            ranks = []
            
            for i, item in enumerate(trending[:10]):
                tokens.append(item.get('symbol', '').upper())
                ranks.append(10 - i)  # Reverse rank to make it proportional to hotness
                
            # Plot horizontal bar chart
            y_pos = np.arange(len(tokens))
            
            # Use hotness gradient colors
            cmap = plt.cm.get_cmap('Reds')
            colors = [cmap(i/len(tokens)) for i in range(len(tokens))]
            
            ax.barh(y_pos, ranks, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tokens)
            ax.set_title('Trending Tokens')
            ax.set_xlabel('Trending Score')
            
            # Add labels
            for i, v in enumerate(ranks):
                ax.text(v + 0.1, i, f"{v:.1f}", va='center')
                
        except Exception as e:
            logger.error(f"Error plotting trending tokens chart: {str(e)}")
            ax.text(0.5, 0.5, 'Data unavailable', ha='center')
            
    def create_technical_chart(self, symbol, indicators=None):
        """Create technical analysis chart for a single asset"""
        try:
            # Get price data
            df = self._get_price_data(symbol)
            
            if df is None or df.empty:
                logger.error(f"Unable to get price data for {symbol}")
                return None
                
            # Create candlestick chart
            fig = plt.figure(figsize=(12, 8))
            
            # Create subplot grid
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            
            # Plot OHLC
            self._plot_ohlc(ax1, df)
            
            # Plot volume
            self._plot_volume(ax2, df)
            
            # Plot RSI
            self._plot_rsi(ax3, df)
            
            # If technical indicators are available, add to chart
            if indicators:
                self._add_indicators_to_chart(ax1, df, indicators)
                
            # Add title
            fig.suptitle(f'{symbol} Technical Analysis', fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating technical chart: {str(e)}")
            return None
    
    def _plot_ohlc(self, ax, df):
        """Plot OHLC candlestick chart"""
        try:
            # Get price data for plotting
            dates = df.index
            opens = df['Open'].values
            highs = df['High'].values
            lows = df['Low'].values
            closes = df['Close'].values
            
            # Plot candlesticks
            width = 0.6
            width2 = width/2
            
            up = closes > opens
            down = opens > closes
            
            # Plot candles
            ax.bar(dates[up], height=closes[up]-opens[up], bottom=opens[up], width=width, color='green', alpha=0.5)
            ax.bar(dates[down], height=opens[down]-closes[down], bottom=closes[down], width=width, color='red', alpha=0.5)
            
            # Plot wicks
            ax.bar(dates[up], height=highs[up]-closes[up], bottom=closes[up], width=width2, color='green', alpha=0.5)
            ax.bar(dates[up], height=opens[up]-lows[up], bottom=lows[up], width=width2, color='green', alpha=0.5)
            ax.bar(dates[down], height=highs[down]-opens[down], bottom=opens[down], width=width2, color='red', alpha=0.5)
            ax.bar(dates[down], height=closes[down]-lows[down], bottom=lows[down], width=width2, color='red', alpha=0.5)
            
            # Set labels
            ax.set_ylabel('Price (USD)')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.get_xticklabels(), visible=False)
            
        except Exception as e:
            logger.error(f"Error plotting OHLC chart: {str(e)}")
    
    def _plot_volume(self, ax, df):
        """Plot volume bars"""
        try:
            # Plot volume bars
            ax.bar(df.index, df['Volume'], color='blue', alpha=0.3)
            
            # Set labels
            ax.set_ylabel('Volume')
            
            # Format x-axis
            plt.setp(ax.get_xticklabels(), visible=False)
            
        except Exception as e:
            logger.error(f"Error plotting volume chart: {str(e)}")
    
    def _plot_rsi(self, ax, df):
        """Plot RSI indicator"""
        try:
            # Calculate RSI if not already in dataframe
            if 'RSI' not in df.columns:
                # Simple RSI calculation
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            
            # Plot RSI
            ax.plot(df.index, df['RSI'], color='purple')
            
            # Add overbought/oversold lines
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            
            # Set labels and limits
            ax.set_ylabel('RSI')
            ax.set_ylim(0, 100)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
        except Exception as e:
            logger.error(f"Error plotting RSI chart: {str(e)}")
            
    def _get_price_data(self, symbol):
        """Get price data for charting from CryptoCompare API"""
        try:
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {
                'fsym': symbol,
                'tsym': 'USD',
                'limit': 30  # Get 30 days of data
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data['Response'] == 'Success':
                    # Create DataFrame
                    df = pd.DataFrame(data['Data']['Data'])
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Rename columns to match mplfinance requirements
                    df = df.rename(columns={
                        'time': 'Date',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volumefrom': 'Volume'
                    })
                    
                    # Set date index
                    df = df.set_index('Date')
                    
                    return df
                else:
                    logger.error(f"API error: {data['Message']}")
                    return self._generate_mock_price_data(symbol)
            else:
                logger.error(f"API request failed: {response.status_code}")
                return self._generate_mock_price_data(symbol)
                
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {str(e)}")
            return self._generate_mock_price_data(symbol)
    
    def _generate_mock_price_data(self, symbol):
        """Generate mock price data when API fails"""
        try:
            # Create date range for last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate random price data
            base_price = 100  # Default price
            
            # Use more realistic prices for well-known assets
            if symbol == 'BTC':
                base_price = 50000
            elif symbol == 'ETH':
                base_price = 3000
            elif symbol == 'BNB':
                base_price = 400
            elif symbol == 'SOL':
                base_price = 100
            
            # Generate price series with random walk
            np.random.seed(42)  # For reproducibility
            random_walk = np.random.normal(0, 0.02, len(dates))
            price_changes = 1 + np.cumsum(random_walk)
            
            close_prices = base_price * price_changes
            open_prices = close_prices * (1 + np.random.normal(0, 0.01, len(dates)))
            high_prices = np.maximum(close_prices, open_prices) * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
            low_prices = np.minimum(close_prices, open_prices) * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
            volumes = base_price * 1000 * (1 + np.random.normal(0, 0.5, len(dates)))
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': open_prices,
                'High': high_prices,
                'Low': low_prices,
                'Close': close_prices,
                'Volume': volumes
            }, index=dates)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating mock price data: {str(e)}")
            return pd.DataFrame()
            
    def _add_indicators_to_chart(self, ax, df, indicators):
        """Add technical indicators to chart"""
        try:
            # Add moving averages
            if 'ma_50' in indicators:
                ma50 = [indicators['ma_50']] * len(df)
                ax.plot(df.index, ma50, label='MA50', color='blue', linestyle='--')
                
            if 'ma_200' in indicators:
                ma200 = [indicators['ma_200']] * len(df)
                ax.plot(df.index, ma200, label='MA200', color='red', linestyle='--')
                
            # Add Bollinger Bands
            if all(k in indicators for k in ['bb_upper', 'bb_middle', 'bb_lower']):
                bb_upper = [indicators['bb_upper']] * len(df)
                bb_middle = [indicators['bb_middle']] * len(df)
                bb_lower = [indicators['bb_lower']] * len(df)
                
                ax.plot(df.index, bb_upper, label='BB Upper', color='green', alpha=0.5)
                ax.plot(df.index, bb_middle, label='BB Middle', color='blue', alpha=0.5)
                ax.plot(df.index, bb_lower, label='BB Lower', color='green', alpha=0.5)
                
                # Fill Bollinger Bands
                ax.fill_between(df.index, bb_upper, bb_lower, color='green', alpha=0.1)
                
            # Add support/resistance levels
            if 'support_level' in indicators:
                support = [indicators['support_level']] * len(df)
                ax.plot(df.index, support, label='Support', color='green', linestyle='-.')
                
            if 'resistance_level' in indicators:
                resistance = [indicators['resistance_level']] * len(df)
                ax.plot(df.index, resistance, label='Resistance', color='red', linestyle='-.')
                
            # Add signal indicators
            if all(k in indicators for k in ['trend_direction', 'trend_strength']):
                trend_direction = indicators['trend_direction']
                trend_strength = indicators['trend_strength']
                
                # Determine signal text
                signal = "Neutral"
                if trend_direction > 0.3 and trend_strength > 0.5:
                    signal = "Bullish"
                elif trend_direction < -0.3 and trend_strength > 0.5:
                    signal = "Bearish"
                elif trend_direction > 0.3:
                    signal = "Slightly Bullish"
                elif trend_direction < -0.3:
                    signal = "Slightly Bearish"
                
                # Add signal text to the chart
                signal_text = f"Signal: {signal}\n"
                signal_text += f"Strength: {trend_strength:.2f}\n"
                signal_text += f"RSI: {indicators.get('rsi', 0):.1f}\n"
                
                # Add text box
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.05, signal_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=props)
            
            # Add legend
            ax.legend(loc='upper left')
            
        except Exception as e:
            logger.error(f"Error adding technical indicators to chart: {str(e)}")