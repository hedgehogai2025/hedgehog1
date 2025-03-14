import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import logging
import mplfinance as mpf
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ChartGenerator:
    def __init__(self, output_dir: str = "charts"):
        """Initialize the chart generator with an output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set Matplotlib style
        plt.style.use('dark_background')
        self.color_green = '#00cc66'
        self.color_red = '#ff3366'
        self.color_neutral = '#aaaaaa'
        self.color_highlight = '#ffcc00'
        
    def _save_chart(self, filename: str) -> str:
        """Save the current figure to a file and return the path."""
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        return filepath
        
    def generate_market_overview(self, market_data: Dict[str, Any]) -> str:
        """Generate a market overview chart showing multiple metrics."""
        try:
            # Extract top cryptocurrencies by market cap
            top_coins = market_data.get('top_coins', [])[:10]
            
            if not top_coins:
                logger.error("No market data available for overview chart")
                return ""
                
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})
            
            # Top subplot: Market caps and 24h changes
            names = [coin['symbol'].upper() for coin in top_coins]
            
            # Handle cases where market_cap might be missing
            market_caps = []
            for coin in top_coins:
                market_cap = coin.get('market_cap', 0)
                if market_cap is None:
                    market_cap = 0
                market_caps.append(market_cap / 1e9)  # Convert to billions
            
            # Handle cases where price_change_percentage_24h might be missing
            price_changes = []
            for coin in top_coins:
                change = coin.get('price_change_percentage_24h', 0)
                if change is None:
                    change = 0
                price_changes.append(change)
            
            # Color bars based on price change
            colors = [self.color_green if change >= 0 else self.color_red for change in price_changes]
            
            x = np.arange(len(names))
            bar_width = 0.4
            
            # Plot market caps
            bars = ax1.bar(x, market_caps, bar_width, color=colors, alpha=0.8)
            ax1.set_ylabel('Market Cap (Billions USD)')
            ax1.set_title('Top 10 Cryptocurrencies by Market Cap')
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=45)
            
            # Add price change as text on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3, 
                        f"{price_changes[i]:.2f}%", ha='center', va='bottom', 
                        color='white', fontweight='bold')
            
            # Bottom subplot: Market dominance pie chart
            dominance = {coin['symbol'].upper(): coin.get('market_cap', 0) for coin in top_coins}
            total = sum(dominance.values())
            
            # Ensure we have a positive total to avoid division by zero
            if total <= 0:
                logger.warning("Invalid market cap total, using placeholder data")
                dominance = {"BTC": 50, "ETH": 20, "Others": 30}
                total = 100
            else:
                others = market_data.get('total_market_cap', total) - total
                if others > 0:
                    dominance['Others'] = others
                    
            labels = list(dominance.keys())
            sizes = list(dominance.values())
            
            # Custom color map for the pie chart
            pie_colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
            
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=pie_colors)
            ax2.axis('equal')
            ax2.set_title('Market Dominance')
            
            # Add timestamp and data source
            plt.figtext(0.5, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Data: CoinGecko",
                      ha="center", fontsize=8, color=self.color_neutral)
            
            # Save the chart
            filename = f"market_overview_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            return self._save_chart(filename)
            
        except Exception as e:
            logger.error(f"Error generating market overview chart: {str(e)}")
            return ""
            
    def generate_technical_chart(self, 
                               symbol: str, 
                               price_data: pd.DataFrame, 
                               indicators: Dict[str, Any] = None) -> str:
        """
        Generate a technical analysis chart for a cryptocurrency.
        
        Args:
            symbol: The cryptocurrency symbol (e.g., 'BTC')
            price_data: DataFrame with OHLC data (index as DatetimeIndex)
            indicators: Dictionary of technical indicators to overlay
            
        Returns:
            Path to the saved chart or empty string if failed
        """
        try:
            if price_data.empty:
                logger.error(f"No price data available for {symbol}")
                return ""
                
            # Ensure price_data has the right format for mplfinance
            if not isinstance(price_data.index, pd.DatetimeIndex):
                if 'timestamp' in price_data.columns:
                    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                    price_data = price_data.set_index('timestamp')
                else:
                    logger.error(f"Price data for {symbol} doesn't have a proper datetime index")
                    return ""
            
            # Ensure we have OHLC columns
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in price_data.columns:
                    logger.error(f"Missing required column {col} in price data for {symbol}")
                    return ""
            
            # Check if volume column exists, if not, create a synthetic one
            if 'volume' not in price_data.columns:
                logger.warning(f"Volume data missing for {symbol}, creating synthetic volume")
                # Create synthetic volume based on price range
                price_data['volume'] = (price_data['high'] - price_data['low']) * price_data['close'] * 1000
            
            # Rename columns to match mplfinance requirements
            price_data = price_data.rename(columns={
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'volume': 'Volume'
            })
            
            # Apply technical indicators if provided
            if indicators:
                # Example: Add Moving Averages
                if 'sma' in indicators:
                    for period in indicators['sma']:
                        price_data[f'SMA_{period}'] = price_data['Close'].rolling(window=period).mean()
                
                if 'ema' in indicators:
                    for period in indicators['ema']:
                        price_data[f'EMA_{period}'] = price_data['Close'].ewm(span=period, adjust=False).mean()
                        
                # Example: Add RSI
                if 'rsi' in indicators and indicators['rsi']:
                    delta = price_data['Close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    
                    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
                    price_data['RSI'] = 100 - (100 / (1 + rs))
            
            # Setup style and colors for the chart
            mc = mpf.make_marketcolors(
                up=self.color_green,
                down=self.color_red,
                edge='inherit',
                wick='inherit',
                volume='inherit'
            )
            
            s = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='--',
                figcolor='#121212',
                facecolor='#1e1e1e',
                edgecolor='#323232',
                rc={'axes.labelcolor': 'white',
                    'axes.edgecolor': 'gray'}
            )
            
            # Create plot configuration
            plot_config = {
                'type': 'candle',
                'style': s,
                'volume': True,
                'figsize': (12, 8),
                'title': f'{symbol.upper()} Technical Analysis',
                'panel_ratios': (4, 1),
                'datetime_format': '%m-%d',
            }
            
            # Add overlays for technical indicators
            if indicators:
                overlays = []
                
                # Add Moving Averages as overlays
                if 'sma' in indicators:
                    for period in indicators['sma']:
                        if f'SMA_{period}' in price_data.columns:
                            overlays.append(
                                mpf.make_addplot(price_data[f'SMA_{period}'], color='yellow')
                            )
                
                if 'ema' in indicators:
                    for period in indicators['ema']:
                        if f'EMA_{period}' in price_data.columns:
                            overlays.append(
                                mpf.make_addplot(price_data[f'EMA_{period}'], color='cyan')
                            )
                
                # Add RSI in a separate panel
                if 'rsi' in indicators and 'RSI' in price_data.columns:
                    overlays.append(
                        mpf.make_addplot(price_data['RSI'], panel=2, color='magenta', 
                                     ylim=(0, 100), secondary_y=False)
                    )
                    # Update panel ratios to include RSI
                    plot_config['panel_ratios'] = (4, 1, 1)
                
                if overlays:
                    plot_config['addplot'] = overlays
            
            # Save to a file instead of displaying
            filename = f"{symbol.lower()}_technical_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create the plot and save directly
            mpf.plot(price_data, **plot_config, savefig=filepath)
            
            return filepath
        
        except Exception as e:
            logger.error(f"Error generating technical chart for {symbol}: {str(e)}")
            return ""
            
    def generate_sentiment_chart(self, 
                               sentiment_data: Dict[str, Any], 
                               title: str = "Crypto Market Sentiment Analysis") -> str:
        """Generate a chart visualizing sentiment analysis results."""
        try:
            # Extract data
            coins = list(sentiment_data.keys())
            positive_scores = [sentiment_data[coin].get('positive', 0) for coin in coins]
            negative_scores = [sentiment_data[coin].get('negative', 0) for coin in coins]
            neutral_scores = [sentiment_data[coin].get('neutral', 0) for coin in coins]
            
            # Calculate net sentiment
            net_sentiment = [pos - neg for pos, neg in zip(positive_scores, negative_scores)]
            
            # Create figure and axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Bar positions
            x = np.arange(len(coins))
            bar_width = 0.25
            
            # Top subplot: Sentiment breakdown
            ax1.bar(x - bar_width, positive_scores, bar_width, color=self.color_green, label='Positive')
            ax1.bar(x, neutral_scores, bar_width, color=self.color_neutral, label='Neutral')
            ax1.bar(x + bar_width, negative_scores, bar_width, color=self.color_red, label='Negative')
            
            # Add labels and title
            ax1.set_ylabel('Sentiment Score')
            ax1.set_title(title)
            ax1.set_xticks(x)
            ax1.set_xticklabels(coins, rotation=45)
            ax1.legend()
            
            # Bottom subplot: Net sentiment
            colors = [self.color_green if score >= 0 else self.color_red for score in net_sentiment]
            ax2.bar(x, net_sentiment, color=colors)
            ax2.set_ylabel('Net Sentiment')
            ax2.set_xticks(x)
            ax2.set_xticklabels(coins, rotation=45)
            ax2.axhline(y=0, color='gray', linestyle='--')
            
            # Add net sentiment values as text
            for i, score in enumerate(net_sentiment):
                ax2.text(i, score + (0.1 if score >= 0 else -0.1), 
                       f"{score:.2f}", ha='center', va='center' if score >= 0 else 'top', 
                       color='white', fontweight='bold')
            
            # Add timestamp
            plt.figtext(0.5, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      ha="center", fontsize=8, color=self.color_neutral)
            
            # Save chart
            filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            return self._save_chart(filename)
            
        except Exception as e:
            logger.error(f"Error generating sentiment chart: {str(e)}")
            return ""
            
    def generate_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> str:
        """Generate a correlation matrix visualization for crypto assets."""
        try:
            # Create DataFrame from price data dictionary
            df = pd.DataFrame(price_data)
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot heatmap
            cmap = plt.cm.RdYlGn
            im = plt.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im)
            cbar.set_label('Correlation Coefficient')
            
            # Add labels and title
            plt.title('Cryptocurrency Price Correlation Matrix')
            plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
            plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
            
            # Add correlation values in each cell
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
            
            # Save chart
            filename = f"correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            return self._save_chart(filename)
            
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {str(e)}")
            return ""
            
    def generate_sectors_performance(self, sector_data: Dict[str, float]) -> str:
        """Generate a chart showing performance of different crypto sectors."""
        try:
            # Sort sectors by performance
            sorted_sectors = dict(sorted(sector_data.items(), key=lambda x: x[1], reverse=True))
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot horizontal bars
            sectors = list(sorted_sectors.keys())
            performances = list(sorted_sectors.values())
            
            # Color based on performance
            colors = [self.color_green if perf >= 0 else self.color_red for perf in performances]
            
            # Create horizontal bar chart
            bars = plt.barh(sectors, performances, color=colors)
            
            # Add performance values
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label_x_pos = width + 0.5 if width >= 0 else width - 2
                plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{performances[i]:.2f}%', 
                       va='center', color='white', fontweight='bold')
            
            # Add labels and title
            plt.xlabel('24h Performance (%)')
            plt.title('Crypto Sectors Performance')
            plt.axvline(x=0, color='gray', linestyle='--')
            
            # Add timestamp
            plt.figtext(0.5, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      ha="center", fontsize=8, color=self.color_neutral)
            
            # Save chart
            filename = f"sectors_performance_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            return self._save_chart(filename)
            
        except Exception as e:
            logger.error(f"Error generating sectors performance chart: {str(e)}")
            return ""
            
    def generate_trading_signals_chart(self, signals_data: List[Dict[str, Any]]) -> str:
        """Generate a visual representation of trading signals."""
        try:
            if not signals_data:
                logger.error("No signals data provided for chart generation")
                return ""
                
            # Extract data
            symbols = [signal['symbol'] for signal in signals_data]
            signal_types = [signal['signal_type'] for signal in signals_data]  # 'buy' or 'sell'
            strengths = [signal['strength'] for signal in signals_data]  # 0.0 to 1.0
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Assign colors based on signal type
            colors = [self.color_green if signal == 'buy' else self.color_red for signal in signal_types]
            
            # Create horizontal bar chart
            bars = plt.barh(symbols, strengths, color=colors)
            
            # Add strength values and signal type
            for i, bar in enumerate(bars):
                width = bar.get_width()
                # Add strength
                plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                       f'{width:.2f}', 
                       va='center', color='white')
                
                # Add signal type at the beginning of the bar
                plt.text(0.02, bar.get_y() + bar.get_height()/2, 
                       f"{signal_types[i].upper()}", 
                       va='center', ha='left', color='white', 
                       fontweight='bold')
            
            # Add labels and title
            plt.xlabel('Signal Strength')
            plt.title('Crypto Trading Signals')
            plt.xlim(0, 1.1)
            
            # Add timestamp and disclaimer
            plt.figtext(0.5, 0.01, 
                      "DISCLAIMER: For informational purposes only. Not financial advice.",
                      ha="center", fontsize=8, color=self.color_neutral)
            
            # Save chart
            filename = f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            return self._save_chart(filename)
            
        except Exception as e:
            logger.error(f"Error generating trading signals chart: {str(e)}")
            return ""