# 1. Add enhanced error reporting to the post_tweet method in the TwitterClient class:

def post_tweet(self, text):
    """Post a tweet with rate limit handling using v2 API"""
    # If we already know we don't have permission, don't try
    if not self.can_post_tweets:
        logger.info("Skipping tweet posting due to known permission issues")
        return None
        
    max_retries = 3
    retry = 0
    
    while retry < max_retries:
        try:
            if self.rate_limit_manager and not self.rate_limit_manager.can_make_request("tweets"):
                wait_time = 300 + (retry * 300)  # Increased wait time on subsequent retries
                logger.warning(f"Rate limit would be exceeded. Delaying tweet for {wait_time} seconds.")
                time.sleep(wait_time)
                retry += 1
                continue
            
            logger.info(f"Attempting to post tweet via v2 API: {text[:50]}...")
            
            # Use v2 API for tweet posting
            response = self.api_v2.create_tweet(text=text)
            
            # Enhanced logging of the response
            logger.info(f"Twitter API v2 response data: {response.data}")
            
            if self.rate_limit_manager:
                self.rate_limit_manager.register_request("tweets")
            
            logger.info(f"Tweet posted successfully via v2 API: {text[:50]}... with ID: {response.data['id']}")
            
            # Create a response object compatible with the rest of the code
            class TweetResponse:
                def __init__(self, id):
                    self.id = id
            
            return TweetResponse(response.data['id'])
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error posting tweet: {error_msg}")
            
            if "429" in error_msg:
                wait_time = (2 ** retry) * 120  # Increased exponential backoff
                logger.warning(f"Rate limited. Waiting for {wait_time} seconds before retry.")
                time.sleep(wait_time)
                retry += 1
            elif "403" in error_msg:
                logger.error(f"Permission error posting tweet: {error_msg}")
                logger.error(f"Tweet content that failed: {text}")
                self.can_post_tweets = False  # Set flag to avoid future tries
                return None
            else:
                logger.error(f"Unexpected error posting tweet: {error_msg}")
                logger.error(f"Tweet content that failed: {text}")
                return None
    
    logger.error("Max retries reached while posting tweet")
    return None

# 2. Add detailed logging to the content generation in AIXBTStyleContentGenerator:

def generate_market_update(self):
    """Generate market update content"""
    logger.info("Starting market update content generation")
    # Get market data
    market_data = self.crypto_client.get_market_data(count=5)
    global_data = self.crypto_client.get_global_data()
    
    # Check if data was successfully retrieved
    if not market_data or not global_data:
        logger.error("Failed to fetch data for market update")
        return ["Unable to generate market update due to data unavailability. Will retry later. #crypto"]
    
    logger.info(f"Successfully retrieved market data with {len(market_data)} coins")
    
    try:
        # Prepare market summary
        market_summary = {
            "total_market_cap": global_data["data"]["total_market_cap"]["usd"],
            "market_cap_change": global_data["data"]["market_cap_change_percentage_24h_usd"],
            "btc_dominance": global_data["data"]["market_cap_percentage"]["btc"]
        }
        
        logger.info(f"Market summary prepared: Market cap: ${market_summary['total_market_cap']:.2f}, Change: {market_summary['market_cap_change']:.2f}%")
        
        # Prepare top coins data
        top_coins = []
        for coin in market_data:
            top_coins.append({
                "name": coin["name"],
                "symbol": coin["symbol"].upper(),
                "price": coin["current_price"],
                "change_24h": coin["price_change_percentage_24h"],
                "change_7d": coin["price_change_percentage_7d_in_currency"] if "price_change_percentage_7d_in_currency" in coin else None
            })
        
        # Add random focus for variety
        focus_coins = random.sample(top_coins, min(3, len(top_coins)))
        
        logger.info(f"Selected focus coins: {', '.join([coin['symbol'] for coin in focus_coins])}")
        
        # Create prompt for AI
        prompt = f"""
        Generate a Twitter thread (3-4 tweets) with a market update for cryptocurrency. 
        Include the following data:
        
        Global Market:
        - Total Market Cap: ${market_summary['total_market_cap']:.2f}
        - 24h Change: {market_summary['market_cap_change']:.2f}%
        - BTC Dominance: {market_summary['btc_dominance']:.2f}%
        
        Top Performing Coins:
        {', '.join([f"{coin['symbol']}: ${coin['price']:.2f} ({coin['change_24h']:.2f}%)" for coin in focus_coins])}
        
        Format as engaging tweets with relevant hashtags like #crypto #bitcoin. 
        Similar to @aixbt_agent style - professional yet conversational.
        Each tweet should be under 280 characters.
        
        Bot identifier: {self.bot_id}
        """
        
        logger.info("Sending prompt to OpenAI for content generation")
        
        # Generate content
        content = self.ai_client.generate_content(prompt)
        
        if not content:
            logger.error("OpenAI returned empty content")
            return ["Unable to generate market update due to AI service unavailability. #crypto"]
        
        logger.info(f"Successfully generated content from OpenAI: {len(content)} characters")
        logger.info(f"First 100 chars of content: {content[:100]}")
        
        # Split into tweets
        tweets = self._format_as_thread(content)
        logger.info(f"Formatted content into {len(tweets)} tweets")
        
        for i, tweet in enumerate(tweets):
            logger.info(f"Tweet {i+1}: {tweet[:50]}... ({len(tweet)} chars)")
        
        return tweets
        
    except Exception as e:
        logger.error(f"Error generating market update: {str(e)}")
        logger.exception("Full exception details:")
        return ["Experiencing technical difficulties with market updates. Will be back soon! #crypto"]

# 3. Add a manual tweet trigger function at the bottom of run_scheduled_tasks():

def run_scheduled_tasks():
    """Run the bot with scheduled tasks"""
    try:
        # Initialize components
        rate_limit_manager = GlobalRateLimitManager()
        twitter_client = TwitterClient(rate_limit_manager)
        ai_client = OpenAIHelper()  # Using updated OpenAI helper
        crypto_client = CryptoDataClient()
        news_client = NewsDataClient()
        content_generator = AIXBTStyleContentGenerator(ai_client, crypto_client, news_client)
        user_interaction = UserInteraction(twitter_client, ai_client, crypto_client)
        
        # ... [existing code] ...
        
        # Add a diagnostic test tweet function
        def force_tweet_now():
            """Force a tweet to be sent immediately for testing"""
            logger.info("======= DIAGNOSTIC TEST =======")
            logger.info("Manually triggering a test tweet")
            
            # First try a simple direct tweet
            test_content = f"This is a diagnostic test tweet from the updated crypto bot using v2 API. #testing {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            logger.info(f"Attempting to post simple test tweet: {test_content}")
            
            response = twitter_client.post_tweet(test_content)
            
            if response:
                logger.info(f"Test tweet successful with ID: {response.id}")
            else:
                logger.error("Test tweet failed!")
                
                # Try with AI-generated content
                logger.info("Attempting to generate and post AI content...")
                content = content_generator.generate_market_update()
                
                if content and len(content) > 0:
                    logger.info(f"AI generated {len(content)} tweets for the thread")
                    logger.info(f"First tweet content: {content[0]}")
                    
                    # Try posting the first tweet only
                    response = twitter_client.post_tweet(content[0])
                    
                    if response:
                        logger.info(f"AI content tweet successful with ID: {response.id}")
                    else:
                        logger.error("AI content tweet failed!")
                else:
                    logger.error("Failed to generate AI content for testing")
            
            logger.info("======= END DIAGNOSTIC TEST =======")
        
        # Run a diagnostic test tweet immediately
        logger.info("Running diagnostic test in 10 seconds...")
        time.sleep(10)  # Wait 10 seconds to ensure initialization is complete
        force_tweet_now()
        
        # ... [continue with the rest of the scheduling code] ...
        
        logger.info("Advanced Bot started successfully with optimized scheduled tasks")
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes and retry
    except Exception as e:
        logger.error(f"Error initializing bot: {str(e)}")
        raise

# 4. Add a direct test script to verify Twitter API credentials independently:

# Save this as test_twitter_api.py
#!/usr/bin/env python3

import os
import logging
import tweepy
from dotenv import load_dotenv
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Twitter API credentials
TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

def test_twitter_api():
    """Test Twitter API credentials and posting ability"""
    try:
        logger.info("Setting up Twitter API clients")
        
        # Set up v1 client
        auth = tweepy.OAuth1UserHandler(
            TWITTER_CONSUMER_KEY,
            TWITTER_CONSUMER_SECRET,
            TWITTER_ACCESS_TOKEN,
            TWITTER_ACCESS_TOKEN_SECRET
        )
        api_v1 = tweepy.API(auth)
        
        # Set up v2 client
        api_v2 = tweepy.Client(
            consumer_key=TWITTER_CONSUMER_KEY,
            consumer_secret=TWITTER_CONSUMER_SECRET,
            access_token=TWITTER_ACCESS_TOKEN,
            access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
        )
        
        # Verify credentials
        logger.info("Verifying Twitter credentials")
        user = api_v1.verify_credentials()
        logger.info(f"Successfully authenticated as: {user.screen_name}")
        
        # Test posting a tweet via v2 API
        test_tweet = f"Testing Twitter API v2 posting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} #testing"
        logger.info(f"Attempting to post test tweet: {test_tweet}")
        
        response = api_v2.create_tweet(text=test_tweet)
        
        logger.info(f"Tweet posted successfully! Tweet ID: {response.data['id']}")
        logger.info(f"Full response data: {response.data}")
        
        logger.info("Twitter API test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing Twitter API: {str(e)}")
        logger.exception("Full exception details:")

if __name__ == "__main__":
    logger.info("Starting Twitter API test")
    test_twitter_api()