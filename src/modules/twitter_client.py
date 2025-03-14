# modules/twitter_client.py
import os
import logging
import time
import tweepy
from datetime import datetime, timedelta

# 配置日志
logger = logging.getLogger(__name__)

class TwitterClient:
    def __init__(self):
        # Twitter API认证凭据
        self.consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
        self.consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        # 初始化API
        # 初始化API
        self.v1_api = None
        self.v2_api = None
        
        # 初始化API客户端
        self._init_v1_api()
        self._init_v2_api()
        
    def _init_v1_api(self):
        """初始化Twitter API v1.1客户端"""
        try:
            # 设置OAuth 1.0a认证
            auth = tweepy.OAuth1UserHandler(
                self.consumer_key,
                self.consumer_secret,
                self.access_token,
                self.access_token_secret
            )
            
            # 创建API对象
            self.v1_api = tweepy.API(auth)
            
            # 验证凭据
            self.v1_api.verify_credentials()
            logger.info("Twitter API v1.1 initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Twitter API v1.1: {str(e)}")
            self.v1_api = None
            
    def _init_v2_api(self):
        """初始化Twitter API v2客户端"""
        try:
            # 设置OAuth 2.0认证
            self.v2_api = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.consumer_key,
                consumer_secret=self.consumer_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret
            )
            
            logger.info("Twitter API v2 initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Twitter API v2: {str(e)}")
            self.v2_api = None
    
    def post_tweet(self, text, media_paths=None):
        """发布推文，支持媒体附件"""
        if not text:
            logger.error("Cannot post empty tweet")
            return None
            
        try:
            # 优先使用v2 API (更新的)
            if self.v2_api:
                # 处理媒体文件 (如果有)
                media_ids = []
                if media_paths and self.v1_api:
                    for media_path in media_paths:
                        try:
                            if os.path.exists(media_path):
                                media = self.v1_api.media_upload(media_path)
                                media_ids.append(media.media_id)
                                logger.info(f"Media uploaded: {media_path}")
                        except Exception as e:
                            logger.error(f"Error uploading media {media_path}: {str(e)}")
                
                # 发布推文
                if media_ids:
                    response = self.v2_api.create_tweet(
                        text=text,
                        media_ids=media_ids
                    )
                else:
                    response = self.v2_api.create_tweet(
                        text=text
                    )
                
                tweet_id = response.data['id']
                logger.info(f"Posted tweet via v2 API (ID: {tweet_id}): {text[:50]}...")
                return tweet_id
            
            # 回退到v1 API
            elif self.v1_api:
                # 处理媒体文件 (如果有)
                media_ids = []
                if media_paths:
                    for media_path in media_paths:
                        try:
                            if os.path.exists(media_path):
                                media = self.v1_api.media_upload(media_path)
                                media_ids.append(media.media_id)
                                logger.info(f"Media uploaded: {media_path}")
                        except Exception as e:
                            logger.error(f"Error uploading media {media_path}: {str(e)}")
                
                # 发布推文
                if media_ids:
                    status = self.v1_api.update_status(
                        status=text,
                        media_ids=media_ids
                    )
                else:
                    status = self.v1_api.update_status(
                        status=text
                    )
                
                tweet_id = status.id
                logger.info(f"Posted tweet via v1 API (ID: {tweet_id}): {text[:50]}...")
                return tweet_id
            
            else:
                logger.error("No Twitter API client is available")
                return None
                
        except Exception as e:
            logger.error(f"Error posting tweet: {str(e)}")
            return None
            
    def post_tweet_with_media(self, text, media_path):
        """使用媒体发布推文"""
        try:
            # 优先尝试v1 API上传媒体
            if self.v1_api:
                try:
                    media = self.v1_api.media_upload(media_path)
                    status = self.v1_api.update_status(
                        status=text, 
                        media_ids=[media.media_id]
                    )
                    tweet_id = status.id
                    logger.info(f"Posted tweet with media via v1 API (ID: {tweet_id}): {text[:50]}...")
                    return tweet_id
                except Exception as e:
                    logger.error(f"Error posting tweet with media via v1 API: {str(e)}")
            
            # 如果v1 API失败，尝试v2 API (但需要首先上传媒体)
            if self.v2_api and self.v1_api:
                try:
                    media = self.v1_api.media_upload(media_path)
                    response = self.v2_api.create_tweet(
                        text=text,
                        media_ids=[media.media_id]
                    )
                    tweet_id = response.data['id']
                    logger.info(f"Posted tweet with media via v2 API (ID: {tweet_id}): {text[:50]}...")
                    return tweet_id
                except Exception as e:
                    logger.error(f"Error posting tweet with media via v2 API: {str(e)}")
                    
            # 如果都失败了，尝试不带媒体发布
            return self.post_tweet(text)
            
        except Exception as e:
            logger.error(f"Error in post_tweet_with_media: {str(e)}")
            # 尝试不带媒体发布
            return self.post_tweet(text)
            
    def reply_to_tweet(self, tweet_id, text, media_paths=None):
        """回复推文"""
        if not tweet_id or not text:
            logger.error("Cannot reply with empty tweet_id or text")
            return None
            
        try:
            # 优先使用v2 API
            if self.v2_api:
                # 处理媒体文件 (如果有)
                media_ids = []
                if media_paths and self.v1_api:
                    for media_path in media_paths:
                        try:
                            if os.path.exists(media_path):
                                media = self.v1_api.media_upload(media_path)
                                media_ids.append(media.media_id)
                        except Exception as e:
                            logger.error(f"Error uploading media {media_path}: {str(e)}")
                
                # 发布回复
                if media_ids:
                    response = self.v2_api.create_tweet(
                        text=text,
                        media_ids=media_ids,
                        in_reply_to_tweet_id=tweet_id
                    )
                else:
                    response = self.v2_api.create_tweet(
                        text=text,
                        in_reply_to_tweet_id=tweet_id
                    )
                
                reply_id = response.data['id']
                logger.info(f"Replied to tweet {tweet_id} via v2 API (ID: {reply_id}): {text[:50]}...")
                return reply_id
                
            # 回退到v1 API
            elif self.v1_api:
                # 处理媒体文件 (如果有)
                media_ids = []
                if media_paths:
                    for media_path in media_paths:
                        try:
                            if os.path.exists(media_path):
                                media = self.v1_api.media_upload(media_path)
                                media_ids.append(media.media_id)
                        except Exception as e:
                            logger.error(f"Error uploading media {media_path}: {str(e)}")
                
                # 发布回复
                if media_ids:
                    status = self.v1_api.update_status(
                        status=text,
                        in_reply_to_status_id=tweet_id,
                        auto_populate_reply_metadata=True,
                        media_ids=media_ids
                    )
                else:
                    status = self.v1_api.update_status(
                        status=text,
                        in_reply_to_status_id=tweet_id,
                        auto_populate_reply_metadata=True
                    )
                
                reply_id = status.id
                logger.info(f"Replied to tweet {tweet_id} via v1 API (ID: {reply_id}): {text[:50]}...")
                return reply_id
                
            else:
                logger.error("No Twitter API client is available")
                return None
                
        except Exception as e:
            logger.error(f"Error replying to tweet: {str(e)}")
            return None
            
    def create_thread(self, tweets):
        """创建推文线程"""
        if not tweets or len(tweets) == 0:
            logger.error("Cannot create thread with empty tweets")
            return []
            
        thread_ids = []
        previous_tweet_id = None
        
        for i, tweet in enumerate(tweets):
            try:
                # 第一条推文
                if i == 0:
                    tweet_id = self.post_tweet(tweet)
                    if tweet_id:
                        thread_ids.append(tweet_id)
                        previous_tweet_id = tweet_id
                # 回复前一条推文
                else:
                    if previous_tweet_id:
                        reply_id = self.reply_to_tweet(previous_tweet_id, tweet)
                        if reply_id:
                            thread_ids.append(reply_id)
                            previous_tweet_id = reply_id
                
                # 添加短暂延迟避免频率限制
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error creating thread at tweet {i}: {str(e)}")
                
        logger.info(f"Created thread with {len(thread_ids)} tweets")
        return thread_ids
        
    def get_mentions(self, since_id=None, count=20):
        """获取提及"""
        mentions = []
        
        try:
            # 尝试使用v2 API
            if self.v2_api:
                # 默认查询参数
                params = {
                    "expansions": "author_id,referenced_tweets.id",
                    "tweet.fields": "created_at,public_metrics,conversation_id",
                    "user.fields": "username",
                    "max_results": min(count, 100)  # v2 API最多100条
                }
                
                if since_id:
                    params["since_id"] = since_id
                
                response = self.v2_api.get_users_mentions(
                    id=self.v2_api.get_me()[0].data["id"],
                    **params
                )
                
                if response and response.data:
                    mentions = response.data
                    logger.info(f"Fetched {len(mentions)} mentions via v2 API")
                    return mentions
                    
            # 回退到v1 API
            if self.v1_api:
                params = {
                    "count": count,
                    "tweet_mode": "extended"
                }
                
                if since_id:
                    params["since_id"] = since_id
                
                v1_mentions = self.v1_api.mentions_timeline(**params)
                
                if v1_mentions:
                    mentions = v1_mentions
                    logger.info(f"Fetched {len(mentions)} mentions via v1 API")
                    return mentions
                    
            return mentions
            
        except Exception as e:
            logger.error(f"Error fetching mentions: {str(e)}")
            return []
            
    def get_user_tweets(self, username, max_results=10, start_time=None):
        """获取用户的推文"""
        tweets = []
        
        try:
            # 尝试使用v2 API
            if self.v2_api:
                # 先获取用户ID
                user_response = self.v2_api.get_user(username=username)
                if not user_response or not user_response.data:
                    logger.error(f"User {username} not found")
                    return []
                    
                user_id = user_response.data.id
                
                # 设置查询参数
                params = {
                    "max_results": max_results,
                    "tweet.fields": "created_at,public_metrics,text",
                    "expansions": "author_id",
                    "exclude": "retweets,replies"
                }
                
                if start_time:
                    params["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # 获取用户推文
                tweets_response = self.v2_api.get_users_tweets(
                    id=user_id,
                    **params
                )
                
                if tweets_response and tweets_response.data:
                    tweets = tweets_response.data
                    logger.info(f"Fetched {len(tweets)} tweets from {username} via v2 API")
                    return tweets
                    
            # 回退到v1 API
            if self.v1_api:
                params = {
                    "screen_name": username,
                    "count": max_results,
                    "exclude_replies": True,
                    "include_rts": False,
                    "tweet_mode": "extended"
                }
                
                v1_tweets = self.v1_api.user_timeline(**params)
                
                if v1_tweets:
                    # 过滤掉旧的推文
                    if start_time:
                        v1_tweets = [t for t in v1_tweets if t.created_at >= start_time]
                        
                    tweets = v1_tweets
                    logger.info(f"Fetched {len(tweets)} tweets from {username} via v1 API")
                    return tweets
                    
            return tweets
            
        except Exception as e:
            logger.error(f"Error fetching tweets from {username}: {str(e)}")
            return []
            
    def search_recent_tweets(self, query, max_results=100, start_time=None):
        """搜索最近的推文"""
        tweets = []
        
        try:
            # 尝试使用v2 API (更适合搜索)
            if self.v2_api:
                # 设置查询参数
                params = {
                    "query": query,
                    "max_results": min(max_results, 100),  # v2 API每页最多100条
                    "tweet.fields": "created_at,public_metrics,author_id,text",
                    "expansions": "author_id",
                    "user.fields": "username"
                }
                
                if start_time:
                    params["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # 执行搜索
                search_response = self.v2_api.search_recent_tweets(**params)
                
                if search_response and search_response.data:
                    tweets = search_response.data
                    logger.info(f"Found {len(tweets)} tweets matching '{query}' via v2 API")
                    return tweets
                    
            # 回退到v1 API
            if self.v1_api:
                params = {
                    "q": query,
                    "count": min(max_results, 100),
                    "lang": "en",
                    "result_type": "recent",
                    "tweet_mode": "extended"
                }
                
                v1_search = self.v1_api.search_tweets(**params)
                
                if v1_search:
                    # 过滤掉旧的推文
                    if start_time:
                        v1_search = [t for t in v1_search if t.created_at >= start_time]
                        
                    tweets = v1_search
                    logger.info(f"Found {len(tweets)} tweets matching '{query}' via v1 API")
                    return tweets
                    
            return tweets
            
        except Exception as e:
            logger.error(f"Error searching tweets for '{query}': {str(e)}")
            return []
            
    def get_trends(self, woeid=1):
        """获取Twitter趋势"""
        trends = []
        
        try:
            # 使用v1 API获取趋势 (v2 API尚不支持)
            if self.v1_api:
                trends_data = self.v1_api.get_place_trends(woeid)
                
                if trends_data and len(trends_data) > 0:
                    trends = trends_data[0]["trends"]
                    logger.info(f"Fetched {len(trends)} trends via v1 API")
                    return trends
                    
            return trends
            
        except Exception as e:
            logger.error(f"Error fetching trends: {str(e)}")
            return []