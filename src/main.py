import sys
sys.path.append('/home/ubuntu/.local/lib/python3.x/site-packages')  # Replace with your Python version
import os
import time
import json
import logging
import schedule
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import threading
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from io import BytesIO
import base64

# Import custom modules
from modules.twitter_client import TwitterClient
from modules.nlp_analyzer import NLPAnalyzer
from modules.market_data import MarketData
from modules.blockchain_data import BlockchainData
from modules.technical_analysis import TechnicalAnalysis
from modules.chart_generator import ChartGenerator
from modules.defi_monitor import DeFiMonitor
from modules.nft_analyzer import NFTAnalyzer
from modules.trading_signals import TradingSignals
from modules.onchain_analyzer import OnChainAnalyzer
from modules.anomaly_detector import AnomalyDetector
from modules.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

class CryptoAnalysisBot:
    def __init__(self):
        # Initialize all components
        self.twitter = TwitterClient()
        self.nlp = NLPAnalyzer()
        self.market = MarketData()
        self.blockchain = BlockchainData()
        self.technical = TechnicalAnalysis()
        self.chart = ChartGenerator()
        self.defi = DeFiMonitor()
        self.nft = NFTAnalyzer()
        self.trading = TradingSignals()
        self.onchain = OnChainAnalyzer()
        self.anomaly = AnomalyDetector()
        self.report = ReportGenerator()
        
        # KOL list - crypto influencers and analysts
        self.kol_list = [
            'VitalikButerin', 'cz_binance', 'SBF_FTX', 'aantonop', 
            'hasufl', 'ercwl', 'cryptohayes', 'Arthur_0x', 'APompliano', 
            'nntaleb', 'elonmusk', 'CryptoYieldInfo', 'DefiIgnas',
            'RaoulGMI', 'adam3us', 'cburniske', 'TuurDemeester', 'BarrySilbert',
            'ErikVoorhees', 'gladstein', 'novogratz', 'brian_armstrong', 'Excellion'
        ]
        
        # Track last processed mention ID
        self.last_mention_id = self._load_last_mention_id()
        
        # Data storage
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/charts", exist_ok=True)
        os.makedirs(f"{self.data_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.data_dir}/signals", exist_ok=True)
        
        # Tracked top cryptocurrencies
        self.top_assets = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'AVAX', 'MATIC']
        
        # Hot topics and market themes
        self.hot_topics = set()
        self.trending_tokens = {}
        
        # User subscriptions
        self.user_subscriptions = self._load_user_subscriptions()
        
        # Analysis history for anomaly detection and patterns
        self.analysis_history = {}
        
    def _load_last_mention_id(self):
        """Load the last processed mention ID from file"""
        try:
            with open("data/last_mention_id.txt", "r") as f:
                return int(f.read().strip())
        except:
            return None
    
    def _save_last_mention_id(self, mention_id):
        """Save the last processed mention ID to file"""
        with open("data/last_mention_id.txt", "w") as f:
            f.write(str(mention_id))
    
    def _load_user_subscriptions(self):
        """Load user subscription data"""
        try:
            with open(f"{self.data_dir}/user_subscriptions.json", "r") as f:
                return json.load(f)
        except:
            return {}
    
    def _save_user_subscriptions(self):
        """Save user subscription data"""
        with open(f"{self.data_dir}/user_subscriptions.json", "w") as f:
            json.dump(self.user_subscriptions, f, indent=2)
    
    def collect_market_data(self):
        """Collect and analyze current market data"""
        logger.info("Collecting market data...")
        
        # Get top coin data
        top_coins = self.market.get_top_coins(limit=100)
        
        # Get market indicators
        indicators = self.market.calculate_market_indicators(top_coins)
        
        # Get trending coins
        trending = self.market.get_trending_coins()
        
        # Update trending tokens list
        self._update_trending_tokens(trending)
        
        # Get blockchain data
        gas_prices = self.blockchain.get_eth_gas_price()
        whale_txs = self.blockchain.get_whale_transactions()
        defi_stats = self.defi.get_defi_statistics()
        nft_activity = self.nft.get_nft_market_activity()
        
        # Get on-chain data
        onchain_metrics = self.onchain.get_onchain_metrics(['bitcoin', 'ethereum'])
        
        # Generate technical analysis
        technical_indicators = {}
        for asset in self.top_assets:
            technical_indicators[asset] = self.technical.calculate_indicators(asset)
        
        # Generate trading signals
        trading_signals = self.trading.generate_signals(top_coins, technical_indicators)
        
        # Detect market anomalies
        anomalies = self.anomaly.detect_market_anomalies(top_coins, whale_txs)
        
        # Combine all data
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'indicators': indicators,
            'trending_coins': trending,
            'gas_prices': gas_prices,
            'whale_transactions': whale_txs,
            'defi_stats': defi_stats,
            'nft_activity': nft_activity,
            'technical_indicators': technical_indicators,
            'trading_signals': trading_signals,
            'onchain_metrics': onchain_metrics,
            'anomalies': anomalies
        }
        
        # Save data for later analysis
        filename = f"{self.data_dir}/market_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(market_data, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Market data collected and saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving market data: {str(e)}")
        
        # Asynchronously generate market charts
        threading.Thread(target=self._generate_market_charts, args=(market_data,)).start()
        
        # Update analysis history
        self._update_analysis_history(market_data)
        
        return market_data
    
    def _update_trending_tokens(self, trending_data):
        """Update trending tokens list and track growth"""
        now = datetime.now()
        
        # Extract currently trending tokens
        current_trending = {}
        for item in trending_data:
            symbol = item.get('symbol', '').upper()
            if symbol:
                current_trending[symbol] = {
                    'name': item.get('name', ''),
                    'price_btc': item.get('price_btc', 0),
                    'market_cap_rank': item.get('market_cap_rank', 0),
                    'updated_at': now.isoformat()
                }
        
        # Compare with previous data
        new_trending = []
        for symbol, data in current_trending.items():
            if symbol not in self.trending_tokens:
                new_trending.append(symbol)
            elif data['market_cap_rank'] < self.trending_tokens[symbol]['market_cap_rank']:
                # Rank improved
                new_trending.append(symbol)
        
        # Update main list
        self.trending_tokens.update(current_trending)
        
        # Log new trending tokens
        if new_trending:
            logger.info(f"New trending tokens discovered: {', '.join(new_trending)}")
            
            # Save trending token data
            trending_file = f"{self.data_dir}/trending_tokens.json"
            try:
                with open(trending_file, 'w') as f:
                    json.dump(self.trending_tokens, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving trending token data: {str(e)}")
    
    def _generate_market_charts(self, market_data):
        """Generate market data charts"""
        logger.info("Generating market charts...")
        
        try:
            # Generate market overview chart
            overview_chart = self.chart.create_market_overview(market_data)
            if overview_chart:
                overview_path = f"{self.data_dir}/charts/market_overview_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
                overview_chart.savefig(overview_path)
                plt.close(overview_chart)
                logger.info(f"Market overview chart saved to {overview_path}")
            
            # Generate technical analysis charts for top 5 assets
            for asset in self.top_assets[:5]:
                try:
                    # This is the problematic part - check if we have technical indicators for this asset
                    asset_indicators = market_data['technical_indicators'].get(asset, {})
                    if not asset_indicators:
                        logger.warning(f"No technical indicators available for {asset}, skipping chart generation")
                        continue
                        
                    chart = self.chart.create_technical_chart(asset, asset_indicators)
                    if chart and hasattr(chart, 'savefig'):  # Check if it has savefig method
                        chart_path = f"{self.data_dir}/charts/{asset}_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
                        chart.savefig(chart_path)
                        plt.close(chart)
                        logger.info(f"{asset} technical analysis chart saved to {chart_path}")
                    else:
                        logger.warning(f"Invalid chart object for {asset}, cannot save")
                except Exception as e:
                    logger.error(f"Error generating chart for {asset}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error generating market charts: {str(e)}")
    
    def _update_analysis_history(self, market_data):
        """Update historical analysis data for detecting trends and patterns"""
        try:
            # Extract and store key data points
            timestamp = datetime.now()
            key_metrics = {
                'btc_dominance': market_data['indicators'].get('bitcoin_dominance', 0),
                'total_market_cap': market_data['indicators'].get('total_market_cap', 0),
                'avg_sentiment': 0,  # Will be updated during social data collection
                'top_gainers': [x.get('symbol', '') for x in market_data['indicators'].get('top_gainers', [])[:3]],
                'top_losers': [x.get('symbol', '') for x in market_data['indicators'].get('top_losers', [])[:3]],
                'trending': [x.get('symbol', '') for x in market_data['trending_coins'][:3]],
                'whale_activity': len(market_data['whale_transactions'])
            }
            
            # Add to history
            self.analysis_history[timestamp.isoformat()] = key_metrics
            
            # Only keep the last 30 days of history
            cutoff = datetime.now() - timedelta(days=30)
            self.analysis_history = {k: v for k, v in self.analysis_history.items() 
                                   if datetime.fromisoformat(k) > cutoff}
            
            # Save historical data
            with open(f"{self.data_dir}/analysis_history.json", 'w') as f:
                json.dump(self.analysis_history, f, indent=2, cls=NumpyEncoder)
        
        except Exception as e:
            logger.error(f"Error updating analysis history: {str(e)}")
    
    def collect_social_data(self):
        """Collect and analyze social media and news data"""
        logger.info("Collecting social media and news data...")
        
        # Get data from alternative sources (Reddit, Crypto News)
        alternative_data = self.nlp.analyze_alternative_sources()
        
        # Extract sentiment and topics
        avg_compound = alternative_data.get('sentiment', 0)
        topics = alternative_data.get('topics', [])
        
        # Update hot topics
        for topic, count in topics:
            self.hot_topics.add(topic)
        
        # Keep hot topics list at a reasonable size
        if len(self.hot_topics) > 50:
            self.hot_topics = set(list(self.hot_topics)[-50:])
        
        # Integrate social data
        social_data = {
            'timestamp': datetime.now().isoformat(),
            'news_count': alternative_data.get('news_count', 0),
            'reddit_count': alternative_data.get('reddit_count', 0),
            'average_sentiment': avg_compound,
            'topics': topics,
            'sources': alternative_data.get('sources', {}),
            'hot_topics': list(self.hot_topics)
        }
        
        # Update sentiment data in analysis history
        for recent_key in sorted(list(self.analysis_history.keys()))[-5:]:
            self.analysis_history[recent_key]['avg_sentiment'] = avg_compound
        
        # Save data for later analysis
        filename = f"{self.data_dir}/social_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(social_data, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Social data collected and saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving social data: {str(e)}")
        
        return social_data
    
    def generate_market_analysis(self):
        """Generate market analysis based on collected data"""
        logger.info("Generating market analysis...")
        
        try:
            # Load latest data
            market_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('market_data_')])
            social_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('social_data_')])
            
            if not market_files or not social_files:
                logger.warning("No data files found, collecting new data...")
                market_data = self.collect_market_data()
                social_data = self.collect_social_data()
            else:
                with open(f"{self.data_dir}/{market_files[-1]}", 'r') as f:
                    market_data = json.load(f)
                with open(f"{self.data_dir}/{social_files[-1]}", 'r') as f:
                    social_data = json.load(f)
            
            # Extract key information for analysis
            indicators = market_data.get('indicators', {})
            trending = market_data.get('trending_coins', [])
            whale_txs = market_data.get('whale_transactions', [])
            topics = social_data.get('topics', [])
            sentiment = social_data.get('average_sentiment', 0)
            
            # Get news and reddit data
            news_items = social_data.get('sources', {}).get('news', [])[:5]
            reddit_posts = social_data.get('sources', {}).get('reddit', [])[:5]
            
            # Extract trading signals
            trading_signals = market_data.get('trading_signals', {})
            
            # Extract anomalies
            anomalies = market_data.get('anomalies', [])
            
            # Generate market analysis text
            analysis = self.nlp.generate_market_analysis(
                news_articles=news_items, 
                reddit_posts=reddit_posts, 
                topics=topics, 
                overall_sentiment=sentiment,
                trading_signals=trading_signals,
                anomalies=anomalies
            )
            
            # Enrich analysis content, add technical analysis and trading signals
            tech_analysis = self._generate_technical_summary(market_data)
            if tech_analysis:
                analysis += f"\n\n{tech_analysis}"
            
            # Add anomaly detection results
            if anomalies:
                anomaly_text = self._format_anomalies(anomalies)
                analysis += f"\n\n{anomaly_text}"
            
            # Save analysis to local file
            analysis_filename = f"{self.data_dir}/analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            try:
                with open(analysis_filename, 'w') as f:
                    f.write(analysis)
                logger.info(f"Market analysis saved to {analysis_filename}")
            except Exception as e:
                logger.error(f"Error saving analysis to file: {str(e)}")
            
            # Try to post to Twitter
            try:
                # Get market overview chart (if exists)
                chart_files = sorted([f for f in os.listdir(f"{self.data_dir}/charts") 
                                   if f.startswith("market_overview_")])
                chart_path = None
                if chart_files:
                    chart_path = f"{self.data_dir}/charts/{chart_files[-1]}"
                
                # Create main analysis thread
                self._post_analysis_thread(analysis, chart_path)
                
                # Post trading signals and trending coin tracking
                self._post_trading_signals(trading_signals)
                
                # Post anomaly alerts (if any)
                if anomalies:
                    self._post_anomaly_alerts(anomalies)
                
                # New addition: Always post a simple update regardless of strong signals
                self._post_simple_update(market_data, social_data)
                    
            except Exception as e:
                logger.error(f"Error posting to Twitter: {str(e)}")
                logger.info("Analysis generated, but Twitter post failed - this is expected under API limitations")
            
            logger.info("Market analysis successfully generated")
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating market analysis: {str(e)}")
            return "Unable to generate market analysis at this time, please try again later."
    
    def _post_simple_update(self, market_data, social_data):
        """Post a simple market update regardless of strong signals"""
        try:
            # Get basic market data
            indicators = market_data.get('indicators', {})
            trending = market_data.get('trending_coins', [])[:5]
            sentiment_str = "neutral"
            sentiment = social_data.get('average_sentiment', 0)
            
            if sentiment > 0.2:
                sentiment_str = "bullish"
            elif sentiment < -0.2:
                sentiment_str = "bearish"
            
            # Get 24-hour changes for main tokens
            top_coins = {}
            for asset in self.top_assets[:3]:  # Just take top 3 main assets
                for coin in market_data.get('trending_coins', []):
                    if coin.get('symbol', '').upper() == asset:
                        top_coins[asset] = coin.get('price_change_percentage_24h', 0)
            
            # Format simple update
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            update_text = f"📊 #Crypto Market Update ({current_time}):\n\n"
            update_text += f"Total Market Cap: ${indicators.get('total_market_cap', 0)/1e9:.1f}B\n"
            update_text += f"24h Volume: ${indicators.get('total_volume_24h', 0)/1e9:.1f}B\n"
            update_text += f"BTC Dominance: {indicators.get('bitcoin_dominance', 0):.1f}%\n"
            update_text += f"Market Sentiment: {sentiment_str}\n\n"
            
            # Add main token price changes
            update_text += "24h Price Changes:\n"
            for asset, change in top_coins.items():
                emoji = "🟢" if change > 0 else "🔴"
                update_text += f"{emoji} ${asset}: {change:.1f}%\n"
            
            update_text += "\nTrending Tokens: " + ", ".join([f"${t.get('symbol', '')}" for t in trending]) + "\n\n"
            update_text += "#Bitcoin #Ethereum #Blockchain"
            
            # Get market overview chart (if exists)
            chart_files = sorted([f for f in os.listdir(f"{self.data_dir}/charts") 
                               if f.startswith("market_overview_")])
            chart_path = None
            if chart_files:
                chart_path = f"{self.data_dir}/charts/{chart_files[-1]}"
            
            # Post update with chart (if available)
            if chart_path:
                self.twitter.post_tweet_with_media(update_text[:270], chart_path)
            else:
                self.twitter.post_tweet(update_text[:270])
                
            logger.info("Simple market update posted to Twitter")
        except Exception as e:
            logger.error(f"Error posting simple update: {str(e)}")
    
    def _generate_technical_summary(self, market_data):
        """Generate technical analysis summary"""
        try:
            tech_indicators = market_data.get('technical_indicators', {})
            
            # Filter assets with clear signals
            strong_signals = {}
            for asset, indicators in tech_indicators.items():
                if indicators.get('trend_strength', 0) > 0.7:
                    direction = "bullish" if indicators.get('trend_direction', 0) > 0 else "bearish"
                    strong_signals[asset] = {
                        'direction': direction,
                        'strength': indicators.get('trend_strength', 0),
                        'key_level': indicators.get('key_level', 0)
                    }
            
            if not strong_signals:
                return ""
                
            # Format as text
            tech_summary = "📊 **Technical Analysis Summary:**\n"
            for asset, signal in strong_signals.items():
                tech_summary += f"- {asset}: {signal['direction']} (strength: {signal['strength']:.2f}), key level: ${signal['key_level']:,.2f}\n"
                
            return tech_summary
            
        except Exception as e:
            logger.error(f"Error generating technical summary: {str(e)}")
            return ""
    
    def _format_anomalies(self, anomalies):
        """Format anomaly detection results"""
        if not anomalies:
            return ""
            
        anomaly_text = "🚨 **Market Anomaly Alerts:**\n"
        for anomaly in anomalies:
            anomaly_text += f"- {anomaly['asset']}: {anomaly['description']} (confidence: {anomaly['confidence']:.1f}%)\n"
            
        return anomaly_text
    
    def _post_analysis_thread(self, analysis, chart_path=None):
        """Post market analysis as Twitter thread, optionally with chart"""
        logger.info("Posting market analysis to Twitter...")
        
        # If analysis is too long, split into a thread
        if len(analysis) <= 270:
            # Single tweet, possibly with image
            response = self.twitter.post_tweet_with_media(analysis, chart_path) if chart_path else self.twitter.post_tweet(analysis)
            if response:
                # Handle v1 and v2 API responses
                if hasattr(response, 'data') and isinstance(response.data, dict):
                    # v2 API response
                    tweet_id = response.data.get('id')
                    logger.info(f"Analysis successfully posted to Twitter (ID: {tweet_id})")
                elif hasattr(response, 'id'):
                    # v1 API response
                    tweet_id = response.id
                    logger.info(f"Analysis successfully posted to Twitter (ID: {tweet_id})")
                else:
                    logger.info("Analysis successfully posted to Twitter")
        else:
            # Split into thread
            chunks = []
            current_chunk = ""
            for sentence in analysis.split('. '):
                # Add period unless it ends with a punctuation mark
                if not sentence.endswith(('.', '!', '?')):
                    sentence += "."
                
                if len(current_chunk) + len(sentence) + 2 <= 270:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Post thread
            prev_tweet_id = None
            thread_posted = False
            
            for i, chunk in enumerate(chunks):
                # Add thread marker and hashtags to first tweet
                if i == 0:
                    chunk = "🧵 #Crypto Market Analysis:\n\n" + chunk
                
                chunk_text = f"{chunk} ({i+1}/{len(chunks)})"
                
                if i == 0:  # First tweet
                    # First tweet may include image
                    response = self.twitter.post_tweet_with_media(chunk_text, chart_path) if chart_path and i == 0 else self.twitter.post_tweet(chunk_text)
                    
                    if response:
                        thread_posted = True
                        # Handle v1 and v2 API responses
                        if hasattr(response, 'data') and isinstance(response.data, dict):
                            # v2 API response
                            prev_tweet_id = response.data.get('id')
                        elif hasattr(response, 'id'):
                            # v1 API response
                            prev_tweet_id = response.id
                else:  # Reply to previous tweet
                    if prev_tweet_id:
                        response = self.twitter.reply_to_tweet(prev_tweet_id, chunk_text)
                        if response:
                            # Handle v1 and v2 API responses
                            if hasattr(response, 'data') and isinstance(response.data, dict):
                                # v2 API response
                                prev_tweet_id = response.data.get('id')
                            elif hasattr(response, 'id'):
                                # v1 API response
                                prev_tweet_id = response.id
                    else:
                        # If thread is lost, post as standalone tweet
                        self.twitter.post_tweet(chunk_text)
            
            if thread_posted:
                logger.info("Analysis thread successfully posted to Twitter")
    
    def _post_trading_signals(self, trading_signals):
        """Post trading signals"""
        if not trading_signals:
            return
            
        # Select strong signals
        strong_signals = {}
        for asset, signal in trading_signals.items():
            if abs(signal.get('strength', 0)) >= 0.5:  # Lower threshold from 0.7 to 0.5
                strong_signals[asset] = signal
        
        if not strong_signals:
            return
            
        # Format tweet
        signals_text = "📈 #TradingSignal Alert:\n\n"
        for asset, signal in strong_signals.items():
            direction = "🟢 BUY" if signal.get('direction') == 'buy' else "🔴 SELL"
            signals_text += f"{direction} {asset}: Strength {signal.get('strength', 0):.1f}/1.0\n"
            signals_text += f"Entry: ${signal.get('entry_price', 0):,.2f}\n"
            if signal.get('stop_loss'):
                signals_text += f"Stop Loss: ${signal.get('stop_loss', 0):,.2f}\n"
            if signal.get('take_profit'):
                signals_text += f"Take Profit: ${signal.get('take_profit', 0):,.2f}\n"
            signals_text += "\n"
        
        signals_text += "#Crypto #Trading #Investment"
        
        # Post signals
        try:
            self.twitter.post_tweet(signals_text[:270])
            logger.info("Trading signals posted to Twitter")
        except Exception as e:
            logger.error(f"Error posting trading signals: {str(e)}")
    
    def _post_anomaly_alerts(self, anomalies):
        """Post anomaly alerts"""
        if not anomalies:
            return
            
        # Select high confidence anomalies, lower threshold from 80% to 60%
        high_confidence_anomalies = [a for a in anomalies if a.get('confidence', 0) > 60]
        
        if not high_confidence_anomalies:
            return
            
        # Format tweet
        alert_text = "🚨 #MarketAnomaly Alert:\n\n"
        for anomaly in high_confidence_anomalies[:3]:  # Limit to max 3
            alert_text += f"{anomaly['asset']}: {anomaly['description']}\n"
            alert_text += f"Confidence: {anomaly['confidence']:.1f}%\n\n"
        
        alert_text += "Exercise caution with risk management. #Crypto #RiskAlert"
        
        # Post alert
        try:
            self.twitter.post_tweet(alert_text[:270])
            logger.info("Anomaly alerts posted to Twitter")
        except Exception as e:
            logger.error(f"Error posting anomaly alerts: {str(e)}")
    
    def respond_to_mentions(self):
        """Check mentions and respond to user queries"""
        logger.info("Checking mentions...")
        
        try:
            mentions = self.twitter.get_mentions(since_id=self.last_mention_id)
            
            if not mentions:
                logger.info("No new mentions found")
                return
            
            # Determine how to get first mention ID based on response type
            if mentions and len(mentions) > 0:
                # Check if we're dealing with v1 or v2 API response
                first_mention = mentions[0]
                if hasattr(first_mention, 'id'):
                    # v1 API response
                    self.last_mention_id = first_mention.id
                    self._save_last_mention_id(self.last_mention_id)
                elif hasattr(first_mention, 'id_str'):
                    # Another possible v1 format
                    self.last_mention_id = int(first_mention.id_str)
                    self._save_last_mention_id(self.last_mention_id)
                # For v2 API, ID is already in the object
                else:
                    self.last_mention_id = first_mention
                    self._save_last_mention_id(self.last_mention_id)
            
            # Process mentions in chronological order (oldest first)
            for mention in reversed(mentions):
                try:
                    # Extract query from mention text
                    if hasattr(mention, 'full_text'):
                        # v1 API
                        query = mention.full_text.lower()
                        mention_id = mention.id
                        screen_name = mention.user.screen_name
                    elif hasattr(mention, 'text'):
                        # v2 API
                        query = mention.text.lower()
                        mention_id = mention.id
                        # Might not have direct user info in v2
                        screen_name = "user"
                    else:
                        # Unknown format
                        logger.error(f"Unknown mention format: {mention}")
                        continue
                    
                    # Handle commands
                    if "subscribe" in query:
                        self._handle_subscription(query, mention_id, screen_name)
                        continue
                        
                    if "unsubscribe" in query:
                        self._handle_unsubscription(query, mention_id, screen_name)
                        continue
                    
                    if "help" in query:
                        self._send_help_message(mention_id, screen_name)
                        continue
                    
                    # Handle specific asset analysis requests
                    asset_request = self._extract_asset_request(query)
                    if asset_request:
                        self._send_asset_analysis(asset_request, mention_id, screen_name)
                        continue
                    
                    # Check if this is a query we should respond to
                    if any(keyword in query for keyword in ['price', 'analysis', 'thoughts', 'opinion', 'prediction', 'market']):
                        # Extract coin/token names (if present)
                        # This is a simple extraction method, you might want to use named entity recognition for better results
                        tokens = query.split()
                        coin_names = [token for token in tokens if token in self.nlp.crypto_terms]
                        
                        # If relevant data exists, prepare context
                        context = None
                        if coin_names:
                            # Load market data for context
                            market_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('market_data_')])
                            if market_files:
                                with open(f"{self.data_dir}/{market_files[-1]}", 'r') as f:
                                    market_data = json.load(f)
                                
                                # Include relevant data in context
                                context = f"Latest market data about {', '.join(coin_names)}: "
                                for coin in coin_names:
                                    # Look in trending tokens
                                    trending_info = next((item for item in market_data.get('trending_coins', []) 
                                                      if coin.lower() in item.get('name', '').lower() 
                                                      or coin.lower() in item.get('symbol', '').lower()), None)
                                    if trending_info:
                                        context += f"{coin} is trending. "
                                
                        # Generate response
                        response = self.nlp.generate_response_to_query(query, context)
                        
                        # Reply to mention
                        reply_response = self.twitter.reply_to_tweet(mention_id, response)
                        if reply_response:
                            logger.info(f"Replied to mention from @{screen_name}")
                        
                except Exception as e:
                    logger.error(f"Error processing mention: {str(e)}")
        except Exception as e:
            logger.error(f"Error checking mentions: {str(e)}")
            logger.info("Unable to check mentions - this is expected under API limitations")
    
    def _handle_subscription(self, query, mention_id, screen_name):
        """Handle subscription requests"""
        # Extract assets to subscribe to
        tokens = query.split()
        assets = []
        for token in tokens:
            token = token.upper()
            if token in self.top_assets or token in self.trending_tokens:
                assets.append(token)
        
        if not assets:
            self.twitter.reply_to_tweet(mention_id, f"@{screen_name} I couldn't find any valid assets to subscribe to. Please specify valid cryptocurrency tokens like BTC, ETH, etc.")
            return
            
        # Update subscription list
        if screen_name not in self.user_subscriptions:
            self.user_subscriptions[screen_name] = []
            
        for asset in assets:
            if asset not in self.user_subscriptions[screen_name]:
                self.user_subscriptions[screen_name].append(asset)
        
        # Save updated subscriptions
        self._save_user_subscriptions()
        
        # Send confirmation
        assets_list = ", ".join(assets)
        self.twitter.reply_to_tweet(mention_id, f"@{screen_name} You've successfully subscribed to updates for {assets_list}. You'll receive notifications about significant movements.")
        logger.info(f"User {screen_name} subscribed to {assets_list}")
    
    def _handle_unsubscription(self, query, mention_id, screen_name):
        """Handle unsubscription requests"""
        # Check if this is an unsubscribe all
        if "all" in query.lower():
            if screen_name in self.user_subscriptions:
                del self.user_subscriptions[screen_name]
                self._save_user_subscriptions()
                self.twitter.reply_to_tweet(mention_id, f"@{screen_name} You've successfully unsubscribed from all assets.")
                logger.info(f"User {screen_name} unsubscribed from all assets")
            else:
                self.twitter.reply_to_tweet(mention_id, f"@{screen_name} You don't have any active subscriptions currently.")
            return
            
        # Extract assets to unsubscribe from
        tokens = query.split()
        assets = []
        for token in tokens:
            token = token.upper()
            if token in self.top_assets or token in self.trending_tokens:
                assets.append(token)
        
        if not assets or screen_name not in self.user_subscriptions:
            self.twitter.reply_to_tweet(mention_id, f"@{screen_name} I couldn't find any valid assets to unsubscribe from.")
            return
            
        # Update subscription list
        for asset in assets:
            if asset in self.user_subscriptions[screen_name]:
                self.user_subscriptions[screen_name].remove(asset)
        
        # If user has no remaining subscriptions, delete the entire entry
        if not self.user_subscriptions[screen_name]:
            del self.user_subscriptions[screen_name]
        
        # Save updated subscriptions
        self._save_user_subscriptions()
        
        # Send confirmation
        assets_list = ", ".join(assets)
        self.twitter.reply_to_tweet(mention_id, f"@{screen_name} You've successfully unsubscribed from {assets_list}.")
        logger.info(f"User {screen_name} unsubscribed from {assets_list}")
    
    def _send_help_message(self, mention_id, screen_name):
        """Send help information"""
        help_text = f"@{screen_name} Hello! Here's what I can do for you:\n\n"
        help_text += "- Share market analysis and trending assets\n"
        help_text += "- Answer questions about specific tokens\n"
        help_text += "- Provide trading signals and risk alerts\n\n"
        help_text += "Commands:\n"
        help_text += "- subscribe [coin] - Receive alerts for specific tokens\n"
        help_text += "- unsubscribe [coin] - Stop receiving alerts\n"
        help_text += "- analysis [coin] - Get in-depth analysis of a specific token"
        
        self.twitter.reply_to_tweet(mention_id, help_text)
        logger.info(f"Sent help information to user {screen_name}")
    
    def _extract_asset_request(self, query):
        """Extract asset analysis request from query"""
        analysis_keywords = ['analysis', 'report']
        if not any(keyword in query for keyword in analysis_keywords):
            return None
            
        # Extract asset name
        tokens = query.split()
        for token in tokens:
            token = token.upper()
            if token in self.top_assets or token in self.trending_tokens:
                return token
                
        return None
    
    def _send_asset_analysis(self, asset, mention_id, screen_name):
        """Send in-depth analysis for a specific asset"""
        try:
            # Load latest market data
            market_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('market_data_')])
            if not market_files:
                self.twitter.reply_to_tweet(mention_id, f"@{screen_name} Sorry, I currently can't access market data for {asset}. Please try again later.")
                return
                
            with open(f"{self.data_dir}/{market_files[-1]}", 'r') as f:
                market_data = json.load(f)
            
            # Generate asset analysis
            asset_analysis = f"@{screen_name} Analysis for ${asset}:\n\n"
            
            # Technical indicators
            tech_indicators = market_data.get('technical_indicators', {}).get(asset, {})
            if tech_indicators:
                trend = "bullish" if tech_indicators.get('trend_direction', 0) > 0 else "bearish"
                asset_analysis += f"Technical: {trend} (strength: {tech_indicators.get('trend_strength', 0):.2f}/1.0)\n"
                asset_analysis += f"Key level: ${tech_indicators.get('key_level', 0):,.2f}\n\n"
            
            # Trading signals
            signals = market_data.get('trading_signals', {}).get(asset, {})
            if signals:
                direction = "buy" if signals.get('direction') == 'buy' else "sell"
                asset_analysis += f"Signal: {direction} (strength: {signals.get('strength', 0):.2f}/1.0)\n"
                if signals.get('entry_price'):
                    asset_analysis += f"Entry: ${signals.get('entry_price'):,.2f}\n"
                if signals.get('stop_loss'):
                    asset_analysis += f"Stop loss: ${signals.get('stop_loss'):,.2f}\n"
                if signals.get('take_profit'):
                    asset_analysis += f"Take profit: ${signals.get('take_profit'):,.2f}\n\n"
            
            # On-chain data
            onchain = market_data.get('onchain_metrics', {}).get(asset.lower(), {})
            if onchain:
                asset_analysis += "On-chain activity: "
                if onchain.get('active_addresses'):
                    asset_analysis += f"Active addresses: {onchain.get('active_addresses'):,}\n"
                if onchain.get('transaction_volume'):
                    asset_analysis += f"24h volume: ${onchain.get('transaction_volume'):,.2f}\n\n"
            
            # Find asset chart (if exists)
            chart_files = [f for f in os.listdir(f"{self.data_dir}/charts") if f.startswith(f"{asset}_analysis_")]
            chart_path = None
            if chart_files:
                chart_path = f"{self.data_dir}/charts/{sorted(chart_files)[-1]}"  # Get latest chart
            
            # Send analysis, possibly with chart
            if chart_path:
                self.twitter.reply_with_media(mention_id, asset_analysis, chart_path)
            else:
                self.twitter.reply_to_tweet(mention_id, asset_analysis)
                
            logger.info(f"Sent analysis of {asset} to user {screen_name}")
            
        except Exception as e:
            logger.error(f"Error sending asset analysis: {str(e)}")
            self.twitter.reply_to_tweet(mention_id, f"@{screen_name} Sorry, I encountered an issue generating analysis for {asset}. Please try again later.")
    
    def check_signal_alerts(self):
        """Check trading signals and send alerts to subscribers"""
        logger.info("Checking trading signal alerts...")
        
        try:
            # Skip if no subscribers
            if not self.user_subscriptions:
                logger.info("No active user subscriptions")
                return
                
            # Load latest market data
            market_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('market_data_')])
            if not market_files:
                logger.warning("No market data files found")
                return
                
            with open(f"{self.data_dir}/{market_files[-1]}", 'r') as f:
                market_data = json.load(f)
            
            # Get strong trading signals
            trading_signals = market_data.get('trading_signals', {})
            strong_signals = {}
            
            for asset, signal in trading_signals.items():
                if abs(signal.get('strength', 0)) >= 0.6:  # Lower threshold
                    strong_signals[asset] = signal
            
            if not strong_signals:
                logger.info("No strong trading signals")
                return
                
            # Check each user's subscriptions
            for user, subscriptions in self.user_subscriptions.items():
                user_signals = {}
                
                # Filter for user's subscribed assets
                for asset in subscriptions:
                    if asset in strong_signals:
                        user_signals[asset] = strong_signals[asset]
                
                if user_signals:
                    self._send_signal_alert(user, user_signals)
            
        except Exception as e:
            logger.error(f"Error checking signal alerts: {str(e)}")
    
    def _send_signal_alert(self, user, signals):
        """Send trading signal alerts to a specific user"""
        try:
            # Format alert message
            alert_text = f"@{user} New trading signals for your subscribed assets:\n\n"
            
            for asset, signal in signals.items():
                direction = "🟢 BUY" if signal.get('direction') == 'buy' else "🔴 SELL"
                alert_text += f"{direction} ${asset}: Strength {signal.get('strength', 0):.1f}/1.0\n"
                alert_text += f"Entry: ${signal.get('entry_price', 0):,.2f}\n"
                if signal.get('stop_loss'):
                    alert_text += f"Stop loss: ${signal.get('stop_loss', 0):,.2f}\n"
                if signal.get('take_profit'):
                    alert_text += f"Take profit: ${signal.get('take_profit', 0):,.2f}\n\n"
            
            # Send DM or public tweet
            try:
                # Try to send DM
                self.twitter.send_direct_message(user, alert_text)
                logger.info(f"Sent signal alert DM to user {user}")
            except:
                # If DM fails, send public tweet
                self.twitter.post_tweet(alert_text[:270])
                logger.info(f"Sent public signal alert to user {user}")
                
        except Exception as e:
            logger.error(f"Error sending signal alert to {user}: {str(e)}")
    
    def generate_weekly_report(self):
        """Generate weekly market report"""
        logger.info("Generating weekly market report...")
        
        try:
            # Get data for the past week
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            report = self.report.generate_weekly_report(start_date, end_date, self.data_dir)
            
            if not report:
                logger.warning("Unable to generate weekly report, insufficient data")
                return
                
            # Save report
            report_file = f"{self.data_dir}/reports/weekly_report_{end_date.strftime('%Y%m%d')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
                
            logger.info(f"Weekly report saved to {report_file}")
            
            # Post report announcement
            report_announcement = "📊 This week's #Crypto Market Report is now available!\n\n"
            report_announcement += "Includes market overview, key token performance, trending topics, and outlook for next week.\n\n"
            report_announcement += "Click the link below for the full report 👇 #Bitcoin #Blockchain"
            
            # Assuming you have a website to host reports
            # report_url = f"https://yourwebsite.com/reports/weekly_{end_date.strftime('%Y%m%d')}.html"
            # report_announcement += f"\n\n{report_url}"
            
            self.twitter.post_tweet(report_announcement)
            logger.info("Weekly report announcement posted to Twitter")
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {str(e)}")
    
    def schedule_tasks(self):
        """Schedule regular tasks"""
        # Schedule data collection
        schedule.every(2).hours.do(self.collect_market_data)
        schedule.every(1).hour.do(self.collect_social_data)
        
        # Schedule analysis and posting
        schedule.every(4).hours.do(self.generate_market_analysis)
        
        # Schedule mention replies
        schedule.every(10).minutes.do(self.respond_to_mentions)
        
        # Schedule signal alerts
        schedule.every(3).hours.do(self.check_signal_alerts)
        
        # Schedule weekly report (Monday at 9am)
        schedule.every().monday.at("09:00").do(self.generate_weekly_report)
        
        # First run
        self.collect_market_data()
        self.collect_social_data()
        self.generate_market_analysis()
        self.respond_to_mentions()
        
        logger.info("All tasks scheduled, starting main loop")
        
        # Main loop
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every 60 seconds

if __name__ == "__main__":
    logger.info("Starting Cryptocurrency Analysis Bot...")
    bot = CryptoAnalysisBot()
    bot.schedule_tasks()