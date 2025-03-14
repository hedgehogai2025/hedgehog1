#!/usr/bin/env python3
import os
import sys
import time
import traceback
import logging
from datetime import datetime

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/debug_bot.log",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_bot")

def main():
    try:
        logger.info("Starting debug bot")
        logger.info(f"Current directory: {os.getcwd()}")
        
        # Test imports
        logger.info("Testing imports...")
        
        try:
            import tweepy
            logger.info("tweepy imported successfully")
        except Exception as e:
            logger.error(f"Error importing tweepy: {str(e)}")
            return
        
        try:
            import openai
            logger.info("openai imported successfully")
        except Exception as e:
            logger.error(f"Error importing openai: {str(e)}")
            return
        
        try:
            from dotenv import load_dotenv
            logger.info("dotenv imported successfully")
            
            # Test loading environment variables
            load_dotenv()
            twitter_key = os.getenv('TWITTER_CONSUMER_KEY')
            if twitter_key:
                logger.info("Successfully loaded Twitter API key")
            else:
                logger.warning("Twitter API key not found in environment variables")
        except Exception as e:
            logger.error(f"Error with dotenv: {str(e)}")
            return
        
        # Now try importing custom modules
        try:
            sys.path.append(os.path.abspath(os.path.dirname(__file__)))
            from src.modules.twitter_client import TwitterClient
            logger.info("TwitterClient module imported successfully")
            
            from src.modules.nlp_analyzer import NLPAnalyzer
            logger.info("NLPAnalyzer module imported successfully")
            
            from src.modules.market_data import MarketData
            logger.info("MarketData module imported successfully")
            
            from src.modules.blockchain_data import BlockchainData
            logger.info("BlockchainData module imported successfully")
        except Exception as e:
            logger.error(f"Error importing custom modules: {str(e)}")
            logger.error(traceback.format_exc())
            return
        
        # Try initializing components
        logger.info("Initializing components...")
        
        try:
            twitter = TwitterClient()
            logger.info("TwitterClient initialized successfully")
            
            nlp = NLPAnalyzer()
            logger.info("NLPAnalyzer initialized successfully")
            
            market = MarketData()
            logger.info("MarketData initialized successfully")
            
            blockchain = BlockchainData()
            logger.info("BlockchainData initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            logger.error(traceback.format_exc())
            return
        
        logger.info("All components initialized successfully")
        logger.info("Debug bot completed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error in main function: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
    print("Debug bot executed")