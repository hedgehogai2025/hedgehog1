# modules/social_data_collector.py
import os
import logging
import tweepy
import pandas as pd
import json
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)

class SocialDataCollector:
    def __init__(self, twitter_client=None):
        self.twitter_client = twitter_client
        self.kol_list = self._load_kol_list()
        
        # 确保配置目录存在
        os.makedirs('data/config', exist_ok=True)
        os.makedirs('data/social', exist_ok=True)
        
    def _load_kol_list(self):
        """从配置文件加载KOL列表"""
        try:
            kol_file = 'data/config/crypto_kol_list.csv'
            
            # 如果文件不存在，创建默认列表
            if not os.path.exists(kol_file):
                default_kols = [
                    {"username": "VitalikButerin", "category": "Ethereum"},
                    {"username": "cz_binance", "category": "Exchange"},
                    {"username": "SBF_FTX", "category": "Exchange"},
                    {"username": "elonmusk", "category": "Influencer"},
                    {"username": "aantonop", "category": "Bitcoin"},
                    {"username": "CryptoCapo_", "category": "Analysis"},
                    {"username": "PeterLBrandt", "category": "Analysis"},
                    {"username": "saylor", "category": "Bitcoin"},
                    {"username": "zhusu", "category": "Investor"},
                    {"username": "SolanaStatus", "category": "Solana"}
                ]
                
                df = pd.DataFrame(default_kols)
                df.to_csv(kol_file, index=False)
                logger.info(f"Created default KOL list at {kol_file}")
                return [kol["username"] for kol in default_kols]
            
            # 如果文件存在，读取它
            df = pd.read_csv(kol_file)
            logger.info(f"Loaded {len(df)} KOLs from configuration")
            return df['username'].tolist()
        except Exception as e:
            logger.error(f"Error loading KOL list: {str(e)}")
            # 默认KOL列表
            return ["VitalikButerin", "cz_binance", "elonmusk", "aantonop", "CryptoCapo_"]
    
    def collect_kol_tweets(self, hours=24):
        """收集KOL最近的推文"""
        if not self.twitter_client:
            logger.warning("Twitter client not initialized, can't collect KOL tweets")
            return []
            
        all_tweets = []
        since_time = datetime.now() - timedelta(hours=hours)
        
        for kol in self.kol_list:
            try:
                logger.info(f"Collecting tweets from {kol}")
                tweets = self.twitter_client.get_user_tweets(
                    username=kol, 
                    max_results=20,
                    start_time=since_time
                )
                
                if tweets:
                    for tweet in tweets:
                        all_tweets.append({
                            'author': kol,
                            'text': tweet.text if hasattr(tweet, 'text') else tweet['text'],
                            'created_at': tweet.created_at.isoformat() if hasattr(tweet, 'created_at') else tweet['created_at'],
                            'retweet_count': tweet.public_metrics.get('retweet_count', 0) if hasattr(tweet, 'public_metrics') else tweet.get('retweet_count', 0),
                            'like_count': tweet.public_metrics.get('like_count', 0) if hasattr(tweet, 'public_metrics') else tweet.get('like_count', 0),
                            'id': tweet.id if hasattr(tweet, 'id') else tweet['id'],
                            'type': 'kol_tweet'
                        })
                    logger.info(f"Collected {len(tweets)} tweets from {kol}")
                else:
                    logger.info(f"No tweets found for {kol}")
                    
            except Exception as e:
                logger.error(f"Error collecting tweets from {kol}: {str(e)}")
                
        # 保存数据
        if all_tweets:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            with open(f"data/social/kol_tweets_{timestamp}.json", 'w') as f:
                json.dump(all_tweets, f)
                
        return all_tweets
        
    def collect_crypto_trends(self):
        """收集与加密货币相关的Twitter趋势"""
        if not self.twitter_client:
            logger.warning("Twitter client not initialized, can't collect trends")
            return []
            
        try:
            # 获取全球趋势
            trends = self.twitter_client.get_trends()
            
            # 过滤加密相关的趋势
            crypto_keywords = ['crypto', 'bitcoin', 'btc', 'eth', 'blockchain', 'nft', 'defi', 'web3']
            crypto_trends = []
            
            for trend in trends:
                trend_name = trend['name'].lower()
                if any(keyword in trend_name for keyword in crypto_keywords):
                    crypto_trends.append(trend)
                    
            logger.info(f"Found {len(crypto_trends)} crypto-related trends")
            return crypto_trends
            
        except Exception as e:
            logger.error(f"Error collecting crypto trends: {str(e)}")
            return []
            
    def collect_community_sentiment(self, keywords=None):
        """收集社区对特定关键词的情绪"""
        if not self.twitter_client:
            logger.warning("Twitter client not initialized, can't collect community sentiment")
            return []
            
        if not keywords:
            keywords = ["Bitcoin", "Ethereum", "Solana", "Crypto"]
            
        all_tweets = []
        
        for keyword in keywords:
            try:
                logger.info(f"Collecting community tweets about {keyword}")
                tweets = self.twitter_client.search_recent_tweets(
                    query=f"{keyword} -is:retweet", 
                    max_results=100
                )
                
                if tweets:
                    for tweet in tweets:
                        all_tweets.append({
                            'keyword': keyword,
                            'text': tweet.text if hasattr(tweet, 'text') else tweet['text'],
                            'created_at': tweet.created_at.isoformat() if hasattr(tweet, 'created_at') else tweet['created_at'],
                            'author': tweet.author.username if hasattr(tweet, 'author') else tweet.get('author_id', 'unknown'),
                            'retweet_count': tweet.public_metrics.get('retweet_count', 0) if hasattr(tweet, 'public_metrics') else tweet.get('retweet_count', 0),
                            'like_count': tweet.public_metrics.get('like_count', 0) if hasattr(tweet, 'public_metrics') else tweet.get('like_count', 0),
                            'type': 'community_tweet'
                        })
                    
                    logger.info(f"Collected {len(tweets)} community tweets about {keyword}")
                else:
                    logger.info(f"No community tweets found for {keyword}")
                    
            except Exception as e:
                logger.error(f"Error collecting community tweets for {keyword}: {str(e)}")
                
        # 保存数据
        if all_tweets:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            with open(f"data/social/community_tweets_{timestamp}.json", 'w') as f:
                json.dump(all_tweets, f)
                
        return all_tweets