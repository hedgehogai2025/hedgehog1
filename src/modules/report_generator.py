import logging
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/report_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        pass
        
    def generate_weekly_report(self, start_date, end_date, data_dir):
        """Generate weekly report for specified date range"""
        try:
            # Get all data files within the date range
            market_files = self._get_date_range_files(data_dir, 'market_data_', start_date, end_date)
            social_files = self._get_date_range_files(data_dir, 'social_data_', start_date, end_date)
            
            if not market_files or not social_files:
                logger.warning(f"Insufficient data found for period {start_date} to {end_date}")
                return None
                
            # Load market data
            market_data = []
            for file in market_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    data['file_date'] = self._extract_date_from_filename(file)
                    market_data.append(data)
                    
            # Load social data
            social_data = []
            for file in social_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    data['file_date'] = self._extract_date_from_filename(file)
                    social_data.append(data)
                    
            # Generate report content
            report = self._compile_weekly_report(market_data, social_data, start_date, end_date)
            
            # Save report
            report_dir = f"{data_dir}/reports"
            os.makedirs(report_dir, exist_ok=True)
            
            report_file = f"{report_dir}/weekly_report_{end_date.strftime('%Y%m%d')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
                
            logger.info(f"Weekly report generated and saved to {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {str(e)}")
            return None
            
    def _get_date_range_files(self, data_dir, prefix, start_date, end_date):
        """Get data files within specified date range"""
        files = []
        
        try:
            all_files = [f for f in os.listdir(data_dir) if f.startswith(prefix)]
            for file in all_files:
                file_date = self._extract_date_from_filename(file)
                if file_date and start_date <= file_date <= end_date:
                    files.append(os.path.join(data_dir, file))
                    
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Error getting date range files: {str(e)}")
            return []
            
    def _extract_date_from_filename(self, filename):
        """Extract date from filename"""
        try:
            # Assumes filename format is xxx_YYYYMMDD_HHMM.xxx
            basename = os.path.basename(filename)
            parts = basename.split('_')
            
            if len(parts) >= 2:
                date_part = parts[-2]
                if len(date_part) == 8 and date_part.isdigit():
                    return datetime.strptime(date_part, '%Y%m%d')
                    
            return None
            
        except Exception:
            return None
            
    def _compile_weekly_report(self, market_data, social_data, start_date, end_date):
        """Compile weekly report content"""
        try:
            # Get latest market data
            latest_market = market_data[-1] if market_data else {}
            
            # Calculate market changes
            first_market = market_data[0] if market_data else {}
            market_change = self._calculate_market_changes(first_market, latest_market)
            
            # Get hot topics
            hot_topics = self._extract_hot_topics(social_data)
            
            # Get best and worst performing assets
            top_performers, worst_performers = self._get_top_and_worst_performers(market_data)
            
            # Compile report text
            report = f"# Cryptocurrency Market Weekly Report\n\n"
            report += f"Report Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
            
            # Market Overview
            report += "## Market Overview\n\n"
            if market_change:
                btc_dom_change = market_change.get('bitcoin_dominance_change', 0)
                market_cap_change = market_change.get('total_market_cap_change_pct', 0)
                
                report += f"* Total Market Cap: ${latest_market.get('indicators', {}).get('total_market_cap', 0)/1e9:.1f} billion "
                report += f"({market_cap_change:+.1f}%)\n"
                
                report += f"* Bitcoin Dominance: {latest_market.get('indicators', {}).get('bitcoin_dominance', 0):.1f}% "
                report += f"({btc_dom_change:+.1f}%)\n"
                
                report += f"* Ethereum Dominance: {latest_market.get('indicators', {}).get('ethereum_dominance', 0):.1f}%\n\n"
            
            # Best Performers
            report += "## Top Performers This Week\n\n"
            for asset in top_performers:
                report += f"* {asset['symbol'].upper()}: {asset['change']:+.1f}%\n"
            report += "\n"
            
            # Worst Performers
            report += "## Worst Performers This Week\n\n"
            for asset in worst_performers:
                report += f"* {asset['symbol'].upper()}: {asset['change']:+.1f}%\n"
            report += "\n"
            
            # Social Media Hot Topics
            report += "## Hot Topics\n\n"
            for topic, count in hot_topics:
                report += f"* {topic}: {count} mentions\n"
            report += "\n"
            
            # Market Trend Analysis
            report += "## Market Trend Analysis\n\n"
            trend_analysis = self._generate_trend_analysis(market_data, social_data)
            report += trend_analysis + "\n\n"
            
            # Next Week Outlook
            report += "## Next Week Outlook\n\n"
            outlook = self._generate_market_outlook(market_data, social_data)
            report += outlook + "\n\n"
            
            # Disclaimer
            report += "## Disclaimer\n\n"
            report += "This report is for informational purposes only and does not constitute investment advice. Cryptocurrency markets are highly volatile, and investors should act at their own risk.\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error compiling weekly report: {str(e)}")
            return "Unable to generate report: Data processing error"
            
    def _calculate_market_changes(self, first_data, latest_data):
        """Calculate market changes"""
        try:
            first_indicators = first_data.get('indicators', {})
            latest_indicators = latest_data.get('indicators', {})
            
            # Calculate changes
            changes = {}
            
            # Calculate Bitcoin dominance change
            first_btc_dom = first_indicators.get('bitcoin_dominance', 0)
            latest_btc_dom = latest_indicators.get('bitcoin_dominance', 0)
            changes['bitcoin_dominance_change'] = latest_btc_dom - first_btc_dom
            
            # Calculate total market cap change
            first_market_cap = first_indicators.get('total_market_cap', 0)
            latest_market_cap = latest_indicators.get('total_market_cap', 0)
            
            if first_market_cap > 0:
                pct_change = ((latest_market_cap - first_market_cap) / first_market_cap) * 100
                changes['total_market_cap_change_pct'] = pct_change
            else:
                changes['total_market_cap_change_pct'] = 0
                
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating market changes: {str(e)}")
            return {}
            
    def _extract_hot_topics(self, social_data):
        """Extract hot topics from social data"""
        try:
            # Collect all topics and mention counts
            all_topics = {}
            
            for data in social_data:
                topics = data.get('topics', [])
                for topic, count in topics:
                    if topic in all_topics:
                        all_topics[topic] += count
                    else:
                        all_topics[topic] = count
                        
            # Sort by mention count
            sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)
            
            # Return top 10
            return sorted_topics[:10]
            
        except Exception as e:
            logger.error(f"Error extracting hot topics: {str(e)}")
            return []
            
    def _get_top_and_worst_performers(self, market_data):
        """Get best and worst performing assets"""
        try:
            if not market_data:
                return [], []
                
            # Get first and last data points
            first_data = market_data[0]
            last_data = market_data[-1]
            
            # Compare changes for top coins
            first_top_coins = first_data.get('indicators', {}).get('top_gainers', []) + first_data.get('indicators', {}).get('top_losers', [])
            last_top_coins = last_data.get('indicators', {}).get('top_gainers', []) + last_data.get('indicators', {}).get('top_losers', [])
            
            # Create changes dictionary
            changes = {}
            
            # Get prices from first data point
            for coin in first_top_coins:
                symbol = coin.get('symbol', '').lower()
                if symbol:
                    changes[symbol] = {'first_price': coin.get('current_price', 0), 'change': 0}
                    
            # Update prices from last data point and calculate changes
            for coin in last_top_coins:
                symbol = coin.get('symbol', '').lower()
                if symbol in changes and changes[symbol]['first_price'] > 0:
                    last_price = coin.get('current_price', 0)
                    change_pct = ((last_price - changes[symbol]['first_price']) / changes[symbol]['first_price']) * 100
                    changes[symbol]['change'] = change_pct
                    changes[symbol]['symbol'] = symbol
                elif symbol and symbol not in changes:
                    changes[symbol] = {
                        'symbol': symbol,
                        'change': coin.get('price_change_percentage_24h', 0)  # Use 24h change as fallback
                    }
                    
            # Convert to list and sort
            change_list = [{'symbol': k, 'change': v.get('change', 0)} for k, v in changes.items()]
            change_list.sort(key=lambda x: x['change'], reverse=True)
            
            # Get top 5 and bottom 5
            top_performers = change_list[:5]
            worst_performers = change_list[-5:]
            worst_performers.reverse()  # Sort from bad to worse
            
            return top_performers, worst_performers
            
        except Exception as e:
            logger.error(f"Error getting top and worst performers: {str(e)}")
            return [], []
            
    def _generate_trend_analysis(self, market_data, social_data):
        """Generate market trend analysis"""
        try:
            # More complex trend analysis algorithms could be implemented here
            # This is a simplified version as an example
            
            # Check market trend
            market_trend = "neutral"
            sentiment_avg = 0
            
            if market_data:
                latest = market_data[-1]
                indicators = latest.get('indicators', {})
                
                # Check average price change over the last 7 days
                avg_price_change = indicators.get('average_price_change_percentage_24h', 0)
                
                if avg_price_change > 3:
                    market_trend = "strongly bullish"
                elif avg_price_change > 0.5:
                    market_trend = "bullish"
                elif avg_price_change < -3:
                    market_trend = "strongly bearish"
                elif avg_price_change < -0.5:
                    market_trend = "bearish"
                    
            # Social media sentiment
            if social_data:
                # Calculate average sentiment score
                sentiment_sum = sum(data.get('average_sentiment', 0) for data in social_data)
                sentiment_avg = sentiment_sum / len(social_data)
                
            sentiment_desc = "neutral"
            if sentiment_avg > 0.2:
                sentiment_desc = "positive"
            elif sentiment_avg > 0.05:
                sentiment_desc = "slightly positive"
            elif sentiment_avg < -0.2:
                sentiment_desc = "negative"
            elif sentiment_avg < -0.05:
                sentiment_desc = "slightly negative"
                
            # Generate analysis text
            analysis = f"This week the market shows an overall {market_trend} trend. Social media sentiment is generally {sentiment_desc}, "
            
            if market_trend == "bullish" or market_trend == "strongly bullish":
                if sentiment_desc == "positive" or sentiment_desc == "slightly positive":
                    analysis += "with market sentiment aligning with price action, indicating a healthy upward trend."
                else:
                    analysis += "although prices are rising, social media sentiment is lagging, which may indicate unstable upward momentum."
            elif market_trend == "bearish" or market_trend == "strongly bearish":
                if sentiment_desc == "negative" or sentiment_desc == "slightly negative":
                    analysis += "with market sentiment aligning with price action, suggesting the downtrend may continue."
                else:
                    analysis += "despite falling prices, social media sentiment remains positive, potentially signaling an upcoming rebound."
            else:
                analysis += "the market has not yet shown a clear direction and may be in an accumulation phase."
                
            # Add trending tokens
            hot_tokens = []
            if market_data:
                trending = market_data[-1].get('trending_coins', [])
                hot_tokens = [item.get('symbol', '').upper() for item in trending[:3]]
                
            if hot_tokens:
                analysis += f" This week's trending tokens include {', '.join(hot_tokens)}. Investors should monitor developments in these assets."
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {str(e)}")
            return "Unable to generate trend analysis: Insufficient data"
            
    def _generate_market_outlook(self, market_data, social_data):
        """Generate market outlook"""
        try:
            # More complex prediction algorithms could be implemented here
            # This is a simplified version based on recent trends
            
            if not market_data:
                return "Insufficient data to provide an accurate outlook."
                
            # Get latest market data
            latest = market_data[-1]
            indicators = latest.get('indicators', {})
            
            # Generate outlook based on various factors
            btc_dominance = indicators.get('bitcoin_dominance', 0)
            market_volatility = indicators.get('market_volatility_24h', 0)
            
            outlook = ""
            
            # Bitcoin dominance analysis
            if btc_dominance > 60:
                outlook += "Bitcoin dominance is high, which may indicate a conservative market where altcoins could underperform relative to Bitcoin."
            elif btc_dominance < 40:
                outlook += "Bitcoin dominance is low, suggesting investors are more interested in altcoins, which typically occurs in mid-bull markets."
            else:
                outlook += "Bitcoin dominance is at a moderate level, indicating a relatively balanced market."
                
            # Volatility analysis
            if market_volatility > 5:
                outlook += " Market volatility is high, suggesting significant price movements may occur in the coming week. Investors should exercise caution."
            elif market_volatility < 2:
                outlook += " Market volatility is low, possibly indicating energy accumulation that may lead to a directional breakout in the near future."
                
            # Regulatory and macro factors
            outlook += " Investors should closely monitor major central bank policies and regulatory developments, as these factors may influence market sentiment."
            
            # Technical analysis advice
            outlook += " From a technical perspective, key support and resistance levels should be used as indicators for entering and exiting the market."
            
            return outlook
            
        except Exception as e:
            logger.error(f"Error generating market outlook: {str(e)}")
            return "Unable to generate market outlook: Data processing error"