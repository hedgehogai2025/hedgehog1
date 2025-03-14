#!/usr/bin/env python3

import os
import sys
import time
import logging
import json
import schedule
import argparse
from datetime import datetime
import threading
from dotenv import load_dotenv

# Import custom modules
from modules.twitter_client import TwitterClient
from modules.market_data import MarketDataCollector
from modules.nlp_analyzer import NLPAnalyzer
from modules.social_data_collector import SocialDataCollector
from modules.content_generator import ContentGenerator
from modules.chart_generator import ChartGenerator
from modules.user_interaction import UserInteractionHandler
from modules.openai_analyzer import OpenAIAnalyzer  # New import for OpenAI integration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crypto_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Analysis Bot')
    parser.add_argument('--no-post', action='store_true', help='Do not post to Twitter')
    parser.add_argument('--data-dir', default='data', help='Directory to store data files')
    parser.add_argument('--charts-dir', default='charts', help='Directory to store chart images')
    parser.add_argument('--templates-dir', default='templates', help='Directory for templates')
    parser.add_argument('--cache-dir', default='cache', help='Directory for OpenAI cache')
    parser.add_argument('--openai-model', default='gpt-3.5-turbo', help='OpenAI model to use')
    return parser.parse_args()

def check_directories(args):
    """Ensure necessary directories exist."""
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.charts_dir, exist_ok=True)
    os.makedirs(args.templates_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def initialize_clients(args):
    """Initialize API clients and components."""
    # Twitter client
    twitter_client = TwitterClient(
        consumer_key=os.getenv('TWITTER_CONSUMER_KEY'),
        consumer_secret=os.getenv('TWITTER_CONSUMER_SECRET'),
        access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
        access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        bearer_token=os.getenv('TWITTER_BEARER_TOKEN')
    )
    
    # Market data collector
    market_data_collector = MarketDataCollector(
        coingecko_api_key=os.getenv('COINGECKO_API_KEY'),
        cryptocompare_api_key=os.getenv('CRYPTOCOMPARE_API_KEY'),
        data_dir=args.data_dir
    )
    
    # NLP analyzer
    nlp_analyzer = NLPAnalyzer(
        reddit_client_id=os.getenv('REDDIT_CLIENT_ID'),
        reddit_client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        reddit_user_agent='CryptoAnalysisBot/1.0 (by /u/hedgehogai2025)',
        cryptocompare_api_key=os.getenv('CRYPTOCOMPARE_API_KEY'),
        coingecko_api_key=os.getenv('COINGECKO_API_KEY'),
        data_dir=args.data_dir
    )
    
    # OpenAI analyzer
    openai_analyzer = OpenAIAnalyzer(
        api_key=os.getenv('OPENAI_API_KEY'),
        model=args.openai_model,
        cache_dir=args.cache_dir
    )
    
    # Social data collector
    social_data_collector = SocialDataCollector(
        twitter_client=twitter_client,
        data_dir=args.data_dir
    )
    
    # Content generator
    content_generator = ContentGenerator(
        templates_dir=args.templates_dir
    )
    
    # Chart generator
    chart_generator = ChartGenerator(
        output_dir=args.charts_dir
    )
    
    # User interaction handler
    user_interaction_handler = UserInteractionHandler(
        twitter_client=twitter_client,
        data_dir=args.data_dir
    )
    
    return {
        'twitter_client': twitter_client,
        'market_data_collector': market_data_collector,
        'nlp_analyzer': nlp_analyzer,
        'social_data_collector': social_data_collector,
        'content_generator': content_generator,
        'chart_generator': chart_generator,
        'user_interaction_handler': user_interaction_handler,
        'openai_analyzer': openai_analyzer  # Added OpenAI analyzer
    }

def generate_market_updates(components, args):
    """Generate market updates and post to Twitter."""
    logger.info("Generating market updates...")
    
    try:
        # Collect market and social data
        logger.info("Collecting expanded social and market data...")
        market_data = components['market_data_collector'].collect_market_data()
        social_data = components['social_data_collector'].collect_social_data(components['nlp_analyzer'])
        
        # Save analysis to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        analysis_file = os.path.join(args.data_dir, f"analysis_{timestamp}.txt")
        
        # Use OpenAI for enhanced analysis
        try:
            # Get AI market analysis
            logger.info("Generating AI-enhanced market analysis...")
            ai_market_analysis = components['openai_analyzer'].analyze_market_data(market_data)
            
            # Get AI trend predictions
            if 'price_history' in market_data:
                ai_trend_predictions = components['openai_analyzer'].predict_trends(market_data, social_data)
                logger.info("Generated AI trend predictions")
            else:
                ai_trend_predictions = {"full_analysis": "Insufficient historical data for trend prediction"}
                logger.info("Skipped AI trend predictions due to insufficient historical data")
            
            # Get AI social sentiment analysis
            ai_sentiment_analysis = components['openai_analyzer'].analyze_social_sentiment(social_data)
            logger.info("Generated AI social sentiment analysis")
            
            # Generate AI-enhanced market update tweet
            ai_market_update = components['openai_analyzer'].generate_market_update(market_data)
            logger.info("Generated AI-enhanced market update tweet")
            
            # Save AI analysis to file
            ai_analysis_file = os.path.join(args.data_dir, f"ai_analysis_{timestamp}.json")
            with open(ai_analysis_file, 'w') as f:
                json.dump({
                    "market_analysis": ai_market_analysis,
                    "trend_predictions": ai_trend_predictions,
                    "sentiment_analysis": ai_sentiment_analysis
                }, f, indent=2, default=str)
            logger.info(f"AI analysis saved to {ai_analysis_file}")
        except Exception as e:
            logger.error(f"Error generating AI analysis: {str(e)}")
            ai_market_update = None
            ai_market_analysis = None
        
        # Generate charts if market data is available
        if market_data and market_data.get('top_coins'):
            try:
                # Generate market overview chart
                market_chart = components['chart_generator'].generate_market_overview(market_data)
                
                # Generate technical charts for top coins
                btc_data = None
                eth_data = None
                bnb_data = None
                sol_data = None
                xrp_data = None
                
                for coin in market_data.get('top_coins', []):
                    if coin['symbol'].upper() == 'BTC':
                        ohlc_data = components['market_data_collector'].get_ohlc_data(coin['id'])
                        if not ohlc_data.empty:
                            btc_chart = components['chart_generator'].generate_technical_chart(
                                'BTC', ohlc_data, {'sma': [20, 50], 'ema': [12, 26], 'rsi': True}
                            )
                            
                            # Generate AI technical analysis
                            if ai_market_analysis:
                                btc_technical = components['openai_analyzer'].generate_technical_analysis('BTC', ohlc_data)
                                logger.info("Generated AI technical analysis for BTC")
                    
                    elif coin['symbol'].upper() == 'ETH':
                        ohlc_data = components['market_data_collector'].get_ohlc_data(coin['id'])
                        if not ohlc_data.empty:
                            eth_chart = components['chart_generator'].generate_technical_chart(
                                'ETH', ohlc_data, {'sma': [20, 50], 'ema': [12, 26], 'rsi': True}
                            )
                            
                            # Generate AI technical analysis
                            if ai_market_analysis:
                                eth_technical = components['openai_analyzer'].generate_technical_analysis('ETH', ohlc_data)
                                logger.info("Generated AI technical analysis for ETH")
                    
                    elif coin['symbol'].upper() == 'BNB':
                        ohlc_data = components['market_data_collector'].get_ohlc_data(coin['id'])
                        if not ohlc_data.empty:
                            bnb_chart = components['chart_generator'].generate_technical_chart(
                                'BNB', ohlc_data, {'sma': [20, 50], 'ema': [12, 26], 'rsi': True}
                            )
                    
                    elif coin['symbol'].upper() == 'SOL':
                        ohlc_data = components['market_data_collector'].get_ohlc_data(coin['id'])
                        if not ohlc_data.empty:
                            sol_chart = components['chart_generator'].generate_technical_chart(
                                'SOL', ohlc_data, {'sma': [20, 50], 'ema': [12, 26], 'rsi': True}
                            )
                    
                    elif coin['symbol'].upper() == 'XRP':
                        ohlc_data = components['market_data_collector'].get_ohlc_data(coin['id'])
                        if not ohlc_data.empty:
                            xrp_chart = components['chart_generator'].generate_technical_chart(
                                'XRP', ohlc_data, {'sma': [20, 50], 'ema': [12, 26], 'rsi': True}
                            )
            except Exception as e:
                logger.error(f"Error generating chart: {str(e)}")
        
        # Generate content
        try:
            # Generate market update (use AI version if available)
            if ai_market_update:
                market_update = ai_market_update
                logger.info("Using AI-generated market update")
            else:
                market_update = components['content_generator'].generate_market_update(market_data)
                logger.info("Using template-based market update")
            
            # Generate market analysis
            market_analysis = components['content_generator'].generate_market_analysis(market_data, social_data)
            
            # Generate trading signals
            if ai_market_analysis:
                # Generate AI-enhanced trading signals for top coins
                ai_signals = []
                for coin in market_data.get('top_coins', [])[:5]:  # Top 5 coins
                    try:
                        symbol = coin['symbol'].upper()
                        signal = components['openai_analyzer'].generate_trading_signal(
                            symbol, 
                            market_data, 
                            market_data.get('technical_data', {})
                        )
                        ai_signals.append(signal)
                    except Exception as e:
                        logger.error(f"Error generating AI trading signal for {coin['symbol']}: {str(e)}")
                
                if ai_signals:
                    # Use AI signals if available
                    market_data['trading_signals'] = ai_signals
                    logger.info(f"Generated {len(ai_signals)} AI trading signals")
            
            trading_signals = components['content_generator'].generate_trading_signals(
                market_data.get('trading_signals', [])[:5]  # Top 5 signals
            )
            
            # Generate detailed market thread
            market_thread = components['content_generator'].generate_market_thread(
                market_data, 
                market_data.get('technical_data', {})
            )
            
            # Generate sentiment analysis
            sentiment_analysis = components['content_generator'].generate_sentiment_analysis(social_data)
            
            # Write analysis to file
            with open(analysis_file, 'w') as f:
                f.write("=== Market Update ===\n\n")
                f.write(market_update)
                f.write("\n\n=== Market Analysis ===\n\n")
                f.write(market_analysis)
                f.write("\n\n=== Trading Signals ===\n\n")
                f.write(trading_signals)
                f.write("\n\n=== Market Thread ===\n\n")
                f.write('\n\n'.join(market_thread))
                f.write("\n\n=== Sentiment Analysis ===\n\n")
                f.write(sentiment_analysis)
                
                # Add AI analysis if available
                if ai_market_analysis:
                    f.write("\n\n=== AI Market Analysis ===\n\n")
                    f.write(ai_market_analysis.get("analysis_text", "No AI analysis available"))
                
                if 'btc_technical' in locals():
                    f.write("\n\n=== AI BTC Technical Analysis ===\n\n")
                    f.write(btc_technical)
                
                if 'eth_technical' in locals():
                    f.write("\n\n=== AI ETH Technical Analysis ===\n\n")
                    f.write(eth_technical)
            
            logger.info(f"Market analysis saved to {analysis_file}")
            
            # Post to Twitter if enabled
            if not args.no_post:
                logger.info("Posting market analysis to Twitter...")
                
                # Post market thread
                success, thread_ids = components['twitter_client'].create_thread(market_thread)
                if success:
                    logger.info("Analysis thread successfully posted to Twitter")
                else:
                    logger.error("Failed to post analysis thread to Twitter")
                
                # Post trading signals
                signals_tweet_id = components['twitter_client'].post_tweet(trading_signals)
                if signals_tweet_id:
                    logger.info("Trading signals posted to Twitter")
                
                # Post simple market update
                update_tweet_id = components['twitter_client'].post_tweet(market_update)
                if update_tweet_id:
                    logger.info("Simple market update posted to Twitter")
                
                # Post charts with commentary if available
                if 'market_chart' in locals() and market_chart:
                    chart_tweet = "📊 #Crypto Market Overview\n\nSwipe for more analysis and charts. $BTC $ETH #Bitcoin"
                    chart_tweet_id = components['twitter_client'].post_tweet_with_media(chart_tweet, market_chart)
                
                logger.info("Market analysis successfully generated")
        
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in market update process: {str(e)}")

def check_user_interactions(components, args):
    """Check for user interactions and respond."""
    logger.info("Checking for user interactions...")
    
    try:
        # Get latest market and social data files
        market_data_files = sorted([f for f in os.listdir(args.data_dir) if f.startswith('market_data_')])
        social_data_files = sorted([f for f in os.listdir(args.data_dir) if f.startswith('social_data_')])
        
        market_data = {}
        social_data = {}
        
        if market_data_files:
            with open(os.path.join(args.data_dir, market_data_files[-1]), 'r') as f:
                market_data = json.load(f)
        
        if social_data_files:
            with open(os.path.join(args.data_dir, social_data_files[-1]), 'r') as f:
                social_data = json.load(f)
        
        # Check for new mentions directly
        mentions = components['twitter_client'].get_mentions()
        
        if mentions:
            logger.info(f"Found {len(mentions)} new mentions")
            
            # Process each mention with AI-enhanced responses
            for mention in mentions:
                try:
                    mention_id = mention["id"]
                    mention_text = mention["text"]
                    username = mention["username"]
                    
                    logger.info(f"Processing mention from @{username}: {mention_text}")
                    
                    # Generate AI response
                    ai_response = components['openai_analyzer'].generate_response_to_mention(
                        mention_text, username, market_data
                    )
                    
                    # Reply with AI response
                    reply_id = components['twitter_client'].reply_to_tweet(mention_id, ai_response)
                    if reply_id:
                        logger.info(f"Replied to @{username} with AI-generated response")
                    else:
                        logger.error(f"Failed to reply to @{username}")
                        
                except Exception as e:
                    logger.error(f"Error processing mention: {str(e)}")
        else:
            # Fallback to user interaction handler if direct mention access fails
            processed = components['user_interaction_handler'].process_interactions(market_data, social_data)
            
            if processed > 0:
                logger.info(f"Processed {processed} user interactions")
            else:
                logger.info("No user interactions to process")
    
    except Exception as e:
        logger.error(f"Error checking user interactions: {str(e)}")

def main():
    """Main function to run the crypto analysis bot."""
    args = parse_args()
    check_directories(args)
    
    logger.info("Starting crypto analysis bot with OpenAI integration...")
    
    # Initialize components
    components = initialize_clients(args)
    
    # Schedule tasks
    schedule.every(6).hours.do(generate_market_updates, components=components, args=args)
    schedule.every(5).minutes.do(check_user_interactions, components=components, args=args)
    
    # Log OpenAI configuration
    logger.info(f"OpenAI integration enabled with model: {args.openai_model}")
    
    # Run initial market update
    logger.info("All tasks scheduled, starting main loop")
    generate_market_updates(components, args)
    
    # Main loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == "__main__":
    main()