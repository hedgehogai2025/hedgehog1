import logging
import json
import re
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from jinja2 import Template, Environment, FileSystemLoader
import numpy as np

logger = logging.getLogger(__name__)

class ContentGenerator:
    def __init__(self, templates_dir: str = "templates"):
        """Initialize the content generator with templates directory."""
        self.templates_dir = templates_dir
        
        # Create templates directory if it doesn't exist
        os.makedirs(templates_dir, exist_ok=True)
        
        # Create default templates if they don't exist
        self._create_default_templates()
        
        # Initialize Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(templates_dir))
        
        # Register filters
        self._register_filters()
    
    def _create_default_templates(self):
        """Create default templates if they don't exist."""
        default_templates = {
            "market_update.j2": """📊 #Crypto Market Update ({{ timestamp }}):
{% for coin in top_coins[:5] %}
{{ coin.symbol.upper() }}: ${{ coin.current_price | format_price }} ({{ coin.price_change_percentage_24h | format_percent }}% {{ "🟢" if coin.price_change_percentage_24h >= 0 else "🔴" }})
{% endfor %}
Market Cap: ${{ total_market_cap | format_large_number }}B ({{ market_cap_change | format_percent }}% {{ "🟢" if market_cap_change >= 0 else "🔴" }})
24h Volume: ${{ total_volume | format_large_number }}B
BTC Dominance: {{ btc_dominance | format_percent }}%

#Bitcoin #Ethereum #Crypto""",

            "market_analysis.j2": """📊 Weekly Update:
{% for coin in top_coins[:5] %}
#{{ coin.symbol.upper() }}: {{ coin.price_change_percentage_7d | format_percent }}% ({{ "🟢" if coin.price_change_percentage_7d >= 0 else "🔴" }})
{% endfor %}
Volume: {{ "Up" if volume_change > 0 else "Down" }} {{ volume_change|abs|format_percent }}%

Market Sentiment: {{ market_sentiment }}

Key Events:
{{ market_events }}

#Crypto #Bitcoin #Analysis""",

            "trading_signals.j2": """📈 #TradingSignal Alert:
{% for signal in signals %}
{{ "🟢" if signal.signal_type == "buy" else "🔴" }} {{ signal.signal_type|upper }} {{ signal.symbol }}: Strength {{ signal.strength }}/1.0
{% endfor %}

Based on: {{ signal_methodology }}

DISCLAIMER: Not financial advice
#CryptoTrading""",

            "market_thread.j2": [
                """📊 #Crypto Market Analysis - {{ current_date }}

Overall Market:
Total Cap: ${{ total_market_cap | format_large_number }}B ({{ market_cap_change | format_percent }}% {{ "🟢" if market_cap_change >= 0 else "🔴" }})
24h Vol: ${{ total_volume | format_large_number }}B
BTC Dominance: {{ btc_dominance }}%

#Bitcoin #Crypto
🧵👇""",

                """📈 Sector Rotation: Capital flowing from {{ declining_sector }} into {{ rising_sector }}

Top Performers:
{% for coin in top_performers[:3] %}
#{{ coin.symbol }} +{{ coin.price_change_percentage_24h | format_percent }}%
{% endfor %}

Worst Performers:
{% for coin in worst_performers[:3] %}
#{{ coin.symbol }} {{ coin.price_change_percentage_24h | format_percent }}%
{% endfor %}""",

                """🔍 Technical Overview:

#BTC: {{ btc_analysis }}
#ETH: {{ eth_analysis }}
{% if additional_analysis %}
{{ additional_analysis }}
{% endif %}

Key levels to watch:
{{ key_levels }}"""
            ],

            "sentiment_analysis.j2": """🔍 Sentiment Analysis:
{% for coin, sentiment in coin_sentiment.items() %}
#{{ coin }}: {{ sentiment.overall }} ({{ "🟢" if sentiment.overall_score > 0 else "🔴" }} {{ sentiment.overall_score | format_percent }}%)
{% endfor %}

Community Buzz:
{{ community_topics }}

Data from: Reddit, Twitter, News
#CryptoSentiment"""
        }
        
        # Write default templates to files
        for filename, content in default_templates.items():
            filepath = os.path.join(self.templates_dir, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    if isinstance(content, list):
                        f.write('\n\n'.join(content))
                    else:
                        f.write(content)
    
    def _format_price(self, value):
        """Format price with appropriate precision based on value."""
        if value is None:
            return "N/A"
        
        if value < 0.001:
            return f"{value:.8f}"
        elif value < 0.01:
            return f"{value:.6f}"
        elif value < 1:
            return f"{value:.4f}"
        elif value < 1000:
            return f"{value:.2f}"
        else:
            return f"{value:,.2f}"
    
    def _format_percent(self, value):
        """Format percentage values."""
        if value is None:
            return "N/A"
        
        return f"{value:.2f}"
    
    def _format_large_number(self, value):
        """Format large numbers in billions/millions."""
        if value is None:
            return "N/A"
        
        if value >= 1e12:  # Trillion
            return f"{value/1e12:.2f}T"
        elif value >= 1e9:  # Billion
            return f"{value/1e9:.2f}B"
        elif value >= 1e6:  # Million
            return f"{value/1e6:.2f}M"
        else:
            return f"{value:,.0f}"
    
    def _register_filters(self):
        """Register custom filters for Jinja2 templates."""
        self.env.filters['format_price'] = self._format_price
        self.env.filters['format_percent'] = self._format_percent
        self.env.filters['format_large_number'] = self._format_large_number
    
    def generate_market_update(self, market_data: Dict[str, Any]) -> str:
        """Generate a simple market update for Twitter."""
        try:
            template = self.env.get_template("market_update.j2")
            
            # Prepare template variables
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            top_coins = market_data.get('top_coins', [])
            total_market_cap = market_data.get('total_market_cap', 0) / 1e9  # Convert to billions
            total_volume = market_data.get('total_volume', 0) / 1e9  # Convert to billions
            market_cap_change = market_data.get('market_cap_change_percentage_24h', 0)
            
            # Calculate BTC dominance
            btc_market_cap = 0
            for coin in top_coins:
                if coin['symbol'].upper() == 'BTC':
                    btc_market_cap = coin['market_cap']
                    break
            
            btc_dominance = (btc_market_cap / market_data.get('total_market_cap', 1)) * 100
            
            # Render template
            context = {
                'timestamp': timestamp,
                'top_coins': top_coins,
                'total_market_cap': total_market_cap,
                'total_volume': total_volume,
                'market_cap_change': market_cap_change,
                'btc_dominance': btc_dominance
            }
            
            return template.render(**context)
            
        except Exception as e:
            logger.error(f"Error generating market update: {str(e)}")
            return self.generate_fallback_market_update(market_data)
    
    def generate_fallback_market_update(self, market_data: Dict[str, Any]) -> str:
        """Generate a simple market update when primary generation fails."""
        try:
            # Extract basic data that should always be available
            top_coins = market_data.get('top_coins', [])[:3]  # Just top 3 to be safe
            
            update = "📊 #Crypto Market Update:\n\n"
            
            for coin in top_coins:
                symbol = coin.get('symbol', '').upper()
                price = coin.get('current_price', 0)
                change = coin.get('price_change_percentage_24h', 0)
                
                emoji = "🔴" if change < 0 else "🟢"
                update += f"{symbol}: ${price:,.2f} ({change:.2f}%) {emoji}\n"
            
            update += "\n#Bitcoin #Crypto"
            return update
        except Exception as e:
            logger.error(f"Even fallback content generation failed: {str(e)}")
            return "📊 Crypto market data update temporarily unavailable. Please check back soon. #Crypto #Bitcoin"
    
    def generate_market_analysis(self, market_data: Dict[str, Any], social_data: Dict[str, Any]) -> str:
        """Generate a more detailed market analysis."""
        try:
            template = self.env.get_template("market_analysis.j2")
            
            # Prepare template variables
            top_coins = market_data.get('top_coins', [])
            
            # Calculate volume change
            volume_change = market_data.get('volume_change_percentage_24h', 0)
            
            # Determine market sentiment based on social data
            sentiment_scores = []
            for coin_sentiment in social_data.get('sentiment', {}).values():
                sentiment_scores.append(coin_sentiment.get('overall_score', 0))
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            
            if avg_sentiment > 0.3:
                market_sentiment = "Bullish 🚀"
            elif avg_sentiment > 0.1:
                market_sentiment = "Mildly Bullish 📈"
            elif avg_sentiment > -0.1:
                market_sentiment = "Neutral ↔️"
            elif avg_sentiment > -0.3:
                market_sentiment = "Mildly Bearish 📉"
            else:
                market_sentiment = "Bearish 🐻"
            
            # Extract market events from news or social data
            market_events = "No significant events"
            if 'news' in social_data and social_data['news']:
                market_events = "\n".join([f"- {news['title']}" for news in social_data['news'][:3]])
            elif 'additional_news' in market_data and market_data['additional_news']:
                market_events = "\n".join([f"- {news['title']}" for news in market_data['additional_news'][:3]])
            
            # Render template
            context = {
                'top_coins': top_coins,
                'volume_change': volume_change,
                'market_sentiment': market_sentiment,
                'market_events': market_events
            }
            
            return template.render(**context)
            
        except Exception as e:
            logger.error(f"Error generating market analysis: {str(e)}")
            return "📊 Market Analysis: Error generating analysis. Please try again later."
    
    def generate_trading_signals(self, signals: List[Dict[str, Any]]) -> str:
        """Generate trading signals content."""
        try:
            template = self.env.get_template("trading_signals.j2")
            
            # Methodologies to randomly choose from
            methodologies = [
                "Technical indicators (MA, RSI, MACD)",
                "Volume analysis and price action",
                "Support/Resistance breakouts",
                "Chart pattern recognition",
                "Momentum indicator confluence"
            ]
            
            # Render template
            context = {
                'signals': signals,
                'signal_methodology': np.random.choice(methodologies)
            }
            
            return template.render(**context)
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return "📈 Trading Signals: Error generating signals. Please try again later."
    
    def generate_market_thread(self, market_data: Dict[str, Any], technical_data: Dict[str, Any]) -> List[str]:
        """Generate a detailed market thread for Twitter."""
        try:
            # Make sure to register filters before template processing
            self._register_filters()
            
            # Get list of template parts
            template_content = ''
            with open(os.path.join(self.templates_dir, "market_thread.j2"), 'r') as f:
                template_content = f.read()
            
            # Split template into parts for thread tweets
            template_parts = template_content.split('\n\n')
            tweets = []
            
            # Process each template part
            for part in template_parts:
                # Create template with the environment that has filters registered
                template = self.env.from_string(part)
                
                # Current date
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                # Market data
                total_market_cap = market_data.get('total_market_cap', 0) / 1e9  # Convert to billions
                total_volume = market_data.get('total_volume', 0) / 1e9  # Convert to billions
                market_cap_change = market_data.get('market_cap_change_percentage_24h', 0)
                
                # BTC dominance
                btc_market_cap = 0
                for coin in market_data.get('top_coins', []):
                    if coin['symbol'].upper() == 'BTC':
                        btc_market_cap = coin['market_cap']
                        break
                
                btc_dominance = f"{(btc_market_cap / market_data.get('total_market_cap', 1)) * 100:.2f}%"
                
                # Top and worst performers
                sorted_by_change = sorted(
                    market_data.get('top_coins', [])[:20], 
                    key=lambda x: x.get('price_change_percentage_24h', 0),
                    reverse=True
                )
                
                top_performers = sorted_by_change[:5]
                worst_performers = sorted_by_change[-5:]
                
                # Sector analysis
                sectors = {
                    'DeFi': [coin for coin in market_data.get('top_coins', []) if coin.get('category') == 'defi'],
                    'Layer 1': [coin for coin in market_data.get('top_coins', []) if coin.get('category') == 'layer-1'],
                    'Gaming': [coin for coin in market_data.get('top_coins', []) if coin.get('category') == 'gaming'],
                    'Privacy': [coin for coin in market_data.get('top_coins', []) if coin.get('category') == 'privacy'],
                    'Exchange': [coin for coin in market_data.get('top_coins', []) if coin.get('category') == 'exchange']
                }
                
                # Calculate average performance for each sector
                sector_performance = {}
                for sector, coins in sectors.items():
                    if coins:
                        avg_perf = sum(coin.get('price_change_percentage_24h', 0) for coin in coins) / len(coins)
                        sector_performance[sector] = avg_perf
                
                # Find rising and declining sectors
                if sector_performance:
                    rising_sector = max(sector_performance.items(), key=lambda x: x[1])[0]
                    declining_sector = min(sector_performance.items(), key=lambda x: x[1])[0]
                else:
                    rising_sector = "Altcoins"
                    declining_sector = "Major caps"
                
                # Technical analysis
                btc_analysis = technical_data.get('BTC', {}).get('summary', 'Mixed signals')
                eth_analysis = technical_data.get('ETH', {}).get('summary', 'Mixed signals')
                additional_analysis = technical_data.get('additional', '')
                
                # Key levels to watch
                btc_price = next((coin['current_price'] for coin in market_data.get('top_coins', []) 
                                 if coin['symbol'].upper() == 'BTC'), 0)
                eth_price = next((coin['current_price'] for coin in market_data.get('top_coins', []) 
                                 if coin['symbol'].upper() == 'ETH'), 0)
                
                btc_support = round(btc_price * 0.95, -3)  # Round to nearest thousand
                btc_resistance = round(btc_price * 1.05, -3)
                eth_support = round(eth_price * 0.95, -2)  # Round to nearest hundred
                eth_resistance = round(eth_price * 1.05, -2)
                
                key_levels = f"BTC: Support ${btc_support:,.0f} / Resistance ${btc_resistance:,.0f}\nETH: Support ${eth_support:,.0f} / Resistance ${eth_resistance:,.0f}"
                
                # Render template
                context = {
                    'current_date': current_date,
                    'total_market_cap': total_market_cap,
                    'total_volume': total_volume,
                    'market_cap_change': market_cap_change,
                    'btc_dominance': btc_dominance,
                    'top_performers': top_performers,
                    'worst_performers': worst_performers,
                    'rising_sector': rising_sector,
                    'declining_sector': declining_sector,
                    'btc_analysis': btc_analysis,
                    'eth_analysis': eth_analysis,
                    'additional_analysis': additional_analysis,
                    'key_levels': key_levels
                }
                
                try:
                    rendered_content = template.render(**context)
                    if rendered_content.strip():  # Only add non-empty content
                        tweets.append(rendered_content)
                except Exception as template_error:
                    logger.error(f"Error rendering template part: {str(template_error)}")
                    # Skip this part and continue with others
            
            # If no valid tweets were generated, use fallback
            if not tweets:
                return self.generate_fallback_thread()
                
            return tweets
            
        except Exception as e:
            logger.error(f"Error generating market thread: {str(e)}")
            return self.generate_fallback_thread()
    
    def generate_fallback_thread(self) -> List[str]:
        """Generate a minimal thread when complete data is unavailable."""
        return [
            "📊 #Crypto Market Update\n\nMarket data analysis is being refreshed. Full insights will return in our next update. Stay tuned for comprehensive crypto analysis. #Bitcoin #Cryptocurrency",
            "While our systems update, check our previous posts for the latest completed analysis or reply with specific questions about major cryptocurrencies."
        ]
    
    def generate_sentiment_analysis(self, social_data: Dict[str, Any]) -> str:
        """Generate sentiment analysis content."""
        try:
            template = self.env.get_template("sentiment_analysis.j2")
            
            # Extract coin sentiment
            coin_sentiment = {}
            for coin, data in social_data.get('sentiment', {}).items():
                if not data:
                    continue
                
                score = data.get('overall_score', 0)
                
                if score > 0.3:
                    overall = "Very Bullish 🚀"
                elif score > 0.1:
                    overall = "Bullish 📈"
                elif score > -0.1:
                    overall = "Neutral ↔️"
                elif score > -0.3:
                    overall = "Bearish 📉"
                else:
                    overall = "Very Bearish 🐻"
                
                coin_sentiment[coin] = {
                    'overall': overall,
                    'overall_score': score * 100  # Convert to percentage
                }
            
            # Extract community topics
            all_topics = []
            for platform, posts in social_data.get('posts', {}).items():
                for post in posts:
                    if 'topics' in post:
                        all_topics.extend(post['topics'])
            
            # Count topic occurrences
            topic_counts = {}
            for topic in all_topics:
                if topic in topic_counts:
                    topic_counts[topic] += 1
                else:
                    topic_counts[topic] = 1
            
            # Sort by count and take top 5
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            community_topics = "\n".join([f"#{topic} ({count} mentions)" for topic, count in top_topics])
            
            if not community_topics:
                community_topics = "No significant topics detected"
            
            # Render template
            context = {
                'coin_sentiment': coin_sentiment,
                'community_topics': community_topics
            }
            
            return template.render(**context)
            
        except Exception as e:
            logger.error(f"Error generating sentiment analysis: {str(e)}")
            return "🔍 Sentiment Analysis: Error generating analysis. Please try again later."