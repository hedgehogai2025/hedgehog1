import logging
import json
import os
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SocialDataCollector:
    def __init__(self, twitter_client=None, data_dir: str = "data"):
        """Initialize social data collector with Twitter client and data directory."""
        self.twitter_client = twitter_client
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # List of influential crypto Twitter accounts to track
        self.influential_accounts = [
            "VitalikButerin",    # Ethereum co-founder
            "cz_binance",        # Binance CEO
            "SBF_FTX",           # FTX founder (for historical data)
            "elonmusk",          # Occasionally tweets about crypto
            "aantonop",          # Andreas Antonopoulos
            "CryptoCapo_",       # Crypto analyst
            "PeterLBrandt",      # Trader
            "saylor",            # MicroStrategy CEO
            "zhusu",             # Three Arrows Capital founder
            "SolanaStatus",      # Solana updates
        ]
        
        # Crypto topics to track
        self.crypto_topics = [
            "Bitcoin",
            "Ethereum",
            "Solana",
            "Crypto"
        ]
    
    def collect_influencer_tweets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect tweets from influential crypto accounts."""
        if not self.twitter_client:
            logger.error("Twitter client not initialized")
            return {}
        
        influencer_tweets = {}
        
        # Fetch tweets from each influencer
        for username in self.influential_accounts:
            logger.info(f"Collecting tweets from {username}")
            
            tweets = self.twitter_client.get_user_tweets(username, count=10)
            
            if tweets:
                influencer_tweets[username] = tweets
                logger.info(f"Collected {len(tweets)} tweets from {username}")
            else:
                logger.info(f"No tweets found for {username}")
        
        return influencer_tweets
    
    def collect_community_tweets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect community tweets about specific crypto topics."""
        if not self.twitter_client:
            logger.error("Twitter client not initialized")
            return {}
        
        community_tweets = {}
        
        # Fetch tweets for each topic
        for topic in self.crypto_topics:
            logger.info(f"Collecting community tweets about {topic}")
            
            # Search for tweets about the topic, excluding retweets
            query = f"{topic} -is:retweet"
            tweets = self.twitter_client.search_tweets(query, count=20)
            
            if tweets:
                community_tweets[topic] = tweets
                logger.info(f"Collected {len(tweets)} community tweets for {topic}")
            else:
                logger.info(f"No community tweets found for {topic}")
        
        return community_tweets
    
    def collect_social_data(self, nlp_analyzer=None) -> Dict[str, Any]:
        """Collect comprehensive social data from various sources."""
        try:
            social_data = {}
            
            # Collect tweets from influential accounts
            influencer_tweets = self.collect_influencer_tweets()
            social_data["influencer_tweets"] = influencer_tweets
            
            # Collect community tweets
            community_tweets = self.collect_community_tweets()
            social_data["community_tweets"] = community_tweets
            
            # Incorporate NLP analysis data if available
            if nlp_analyzer:
                nlp_data = nlp_analyzer.collect_social_data()
                
                # Merge NLP data with Twitter data
                social_data["reddit_posts"] = nlp_data.get("reddit_posts", [])
                social_data["news"] = nlp_data.get("news", [])
                social_data["sentiment"] = nlp_data.get("sentiment", {})
                social_data["topics"] = nlp_data.get("topics", {})
            
            # Save data to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = os.path.join(self.data_dir, f"social_data_{timestamp}.json")
            
            with open(filename, "w") as f:
                json.dump(social_data, f, default=str)
            
            logger.info(f"Social data collected and saved to {filename}")
            
            return social_data
        except Exception as e:
            logger.error(f"Error collecting social data: {str(e)}")
            return {}