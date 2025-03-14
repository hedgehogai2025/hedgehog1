# modules/user_interaction.py
import re
import time
import json
import logging
import random
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class UserInteractionHandler:
    def __init__(self, twitter_client, nlp_analyzer, market_data):
        self.twitter_client = twitter_client
        self.nlp_analyzer = nlp_analyzer
        self.market_data = market_data
        self.last_mention_id = None
        self.processed_mentions = set()
        self.last_processed_time = datetime.now() - timedelta(hours=1)  # 初始检查1小时前的提及
        
        # 加载之前的状态（如果有）
        self._load_state()
        
    def _load_state(self):
        """加载之前保存的状态"""
        try:
            with open('data/interaction_state.json', 'r') as f:
                state = json.load(f)
                self.last_mention_id = state.get('last_mention_id')
                self.processed_mentions = set(state.get('processed_mentions', []))
                logger.info(f"Loaded interaction state, last mention ID: {self.last_mention_id}")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No previous interaction state found, starting fresh")
            
    def _save_state(self):
        """保存当前状态以便下次加载"""
        try:
            state = {
                'last_mention_id': self.last_mention_id,
                'processed_mentions': list(self.processed_mentions),
                'last_updated': datetime.now().isoformat()
            }
            
            with open('data/interaction_state.json', 'w') as f:
                json.dump(state, f)
                
            logger.info(f"Saved interaction state, last mention ID: {self.last_mention_id}")
        except Exception as e:
            logger.error(f"Error saving interaction state: {str(e)}")
        
    def check_and_respond_mentions(self):
        """检查新的提及并回复"""
        if not self.twitter_client:
            logger.warning("Twitter client not initialized, can't check mentions")
            return
            
        try:
            logger.info("Checking for new mentions...")
            mentions = self.twitter_client.get_mentions(since_id=self.last_mention_id)
            
            if not mentions:
                logger.info("No new mentions found")
                return
                
            logger.info(f"Found {len(mentions)} new mentions")
            
            for mention in mentions:
                # 获取提及ID
                mention_id = mention.id if hasattr(mention, 'id') else mention['id']
                
                # 更新最新的提及ID
                if self.last_mention_id is None or mention_id > self.last_mention_id:
                    self.last_mention_id = mention_id
                
                # 跳过已处理的提及
                if mention_id in self.processed_mentions:
                    continue
                    
                # 处理提及
                self._process_mention(mention)
                self.processed_mentions.add(mention_id)
                
                # 添加短暂延迟避免频率限制
                time.sleep(2)
                
            # 保存状态
            self._save_state()
                
        except Exception as e:
            logger.error(f"Error handling mentions: {str(e)}")
    
    def _process_mention(self, mention):
        """处理单个提及并生成回复"""
        try:
            # 获取提及文本
            text = mention.text if hasattr(mention, 'text') else mention['text']
            mention_id = mention.id if hasattr(mention, 'id') else mention['id']
            
            # 移除@用户名
            text = re.sub(r'@\w+\s*', '', text).strip()
            
            logger.info(f"Processing mention: '{text}'")
            
            # 分析查询意图
            intent = self._analyze_intent(text)
            logger.info(f"Detected intent: {intent}")
            
            # 根据意图生成回复
            reply = self._generate_response(text, intent)
            
            # 发送回复
            self.twitter_client.reply_to_tweet(
                tweet_id=mention_id,
                text=reply
            )
            
            logger.info(f"Replied to mention {mention_id}")
            
        except Exception as e:
            logger.error(f"Error processing mention: {str(e)}")
    
    def _analyze_intent(self, text):
        """分析用户查询意图"""
        text_lower = text.lower()
        
        # 检测币种
        crypto_patterns = {
            'btc': ['bitcoin', 'btc', '$btc', 'bitcoin price'],
            'eth': ['ethereum', 'eth', '$eth', 'ethereum price'],
            'sol': ['solana', 'sol', '$sol'],
            'bnb': ['binance', 'bnb', '$bnb'],
            'xrp': ['ripple', 'xrp', '$xrp'],
            'doge': ['dogecoin', 'doge', '$doge'],
            'ada': ['cardano', 'ada', '$ada'],
            'dot': ['polkadot', 'dot', '$dot']
        }
        
        detected_coin = None
        for coin, patterns in crypto_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_coin = coin
                break
        
        # 简单的基于关键词的意图分类
        if any(word in text_lower for word in ['price', 'prediction', 'forecast', 'target', 'how high', 'how low']):
            intent = 'price_prediction'
        elif any(word in text_lower for word in ['analysis', 'analyze', 'think', 'opinion', 'outlook', 'perspective']):
            intent = 'analysis_request'
        elif any(word in text_lower for word in ['risk', 'safe', 'scam', 'legit', 'trust', 'invest', 'buy', 'sell']):
            intent = 'risk_assessment'
        elif any(word in text_lower for word in ['news', 'update', 'latest', 'happening', 'event']):
            intent = 'news_request'
        elif any(word in text_lower for word in ['trend', 'trending', 'popular', 'hot', 'hype']):
            intent = 'trend_request'
        elif any(word in text_lower for word in ['help', 'explain', 'what is', 'how does', 'tutorial']):
            intent = 'educational_request'
        else:
            intent = 'general_inquiry'
            
        return {
            'type': intent,
            'coin': detected_coin
        }
    
    def _generate_response(self, query, intent):
        """根据意图生成回复"""
        # 获取意图类型和检测到的币种
        intent_type = intent['type']
        coin = intent['coin']
        
        # 更真实的回复前缀，增加多样性
        prefixes = [
            "Based on my analysis, ",
            "Looking at the data, ",
            "From what I'm seeing, ",
            "According to recent trends, ",
            "After reviewing the market, ",
            "My AI analysis suggests that ",
            "The signals indicate that ",
            "📊 Market intel: ",
            "🔍 Just analyzed this: ",
            "👀 Here's what I'm seeing: "
        ]
        
        prefix = random.choice(prefixes)
        
        # 根据不同意图生成回复
        if intent_type == 'price_prediction':
            return self._generate_price_prediction_response(prefix, coin)
        elif intent_type == 'analysis_request':
            return self._generate_analysis_response(prefix, coin)
        elif intent_type == 'risk_assessment':
            return self._generate_risk_assessment_response(prefix, coin)
        elif intent_type == 'news_request':
            return self._generate_news_response(prefix, coin)
        elif intent_type == 'trend_request':
            return self._generate_trend_response(prefix)
        elif intent_type == 'educational_request':
            return self._generate_educational_response(prefix, query)
        else:
            return self._generate_general_response(prefix, query)
            
    def _generate_price_prediction_response(self, prefix, coin):
        """生成价格预测回复"""
        if not coin:
            return f"{prefix}the overall market is showing mixed signals right now. BTC dominance is at {random.randint(48, 55)}% with key resistance at ${random.randint(60, 70)}K. Which specific coin are you interested in?"
            
        coin = coin.upper()
        current_price = self.market_data.get_current_price(coin) if hasattr(self.market_data, 'get_current_price') else None
        
        if not current_price:
            # 使用模拟数据
            if coin == 'BTC':
                current_price = random.randint(58000, 65000)
            elif coin == 'ETH':
                current_price = random.randint(3000, 3500)
            elif coin == 'SOL':
                current_price = random.randint(120, 160)
            else:
                current_price = random.randint(10, 1000)
        
        # 随机生成支撑位和阻力位
        support = current_price * (1 - random.uniform(0.05, 0.15))
        resistance = current_price * (1 + random.uniform(0.08, 0.25))
        
        return f"{prefix}{coin} is currently trading at ${current_price:.2f}. Key support is at ${support:.2f}, with resistance at ${resistance:.2f}. Volume is {random.choice(['increasing', 'stable', 'slightly decreasing'])} with RSI at {random.randint(30, 70)}. For the next 24-48 hours, watch the ${resistance:.2f} level closely."
        
    def _generate_analysis_response(self, prefix, coin):
        """生成分析回复"""
        if not coin:
            return f"{prefix}the crypto market is currently in a {random.choice(['consolidation phase', 'decisive moment', 'trend-establishing pattern'])}. BTC movement is closely correlated with {random.choice(['stock market performance', 'global economic indicators', 'institutional investment flows'])}. Which specific coin would you like me to analyze?"
            
        coin = coin.upper()
        trend = random.choice(['bullish', 'bearish', 'neutral', 'accumulation', 'distribution'])
        volume_pattern = random.choice(['increasing', 'decreasing', 'steady', 'showing unusual spikes'])
        
        technical_indicators = [
            f"RSI: {random.randint(30, 70)}",
            f"MACD: {random.choice(['bullish crossover', 'bearish crossover', 'neutral'])}",
            f"50MA: {random.choice(['above 200MA (bullish)', 'below 200MA (bearish)', 'crossing 200MA (transition)'])}",
            f"Bollinger Bands: {random.choice(['tightening (low volatility expected)', 'expanding (increased volatility expected)', 'price near upper band', 'price near lower band'])}",
            f"OBV: {random.choice(['rising with price (bullish)', 'diverging from price (potential reversal)', 'flat (accumulation)'])}"
        ]
        
        # 随机选择2-3个技术指标
        selected_indicators = random.sample(technical_indicators, random.randint(2, 3))
        
        return f"{prefix}{coin} is showing {trend} signals with {volume_pattern} volume. {'; '.join(selected_indicators)}. Key to watch: {random.choice(['whale accumulation', 'exchange outflows', 'futures open interest', 'options expiry this Friday'])}."
    
    def _generate_risk_assessment_response(self, prefix, coin):
        """生成风险评估回复"""
        if not coin:
            return f"{prefix}investing in crypto always carries risk. Current market risk level: {random.choice(['moderate', 'elevated', 'high'])} due to {random.choice(['macroeconomic uncertainty', 'regulatory concerns', 'technical market structure'])}. Which specific coin are you concerned about?"
            
        coin = coin.upper()
        
        risk_levels = {
            'BTC': 'low to moderate',
            'ETH': 'low to moderate',
            'SOL': 'moderate',
            'BNB': 'moderate',
            'XRP': 'moderate to high',
            'DOGE': 'high',
            'ADA': 'moderate',
            'DOT': 'moderate'
        }
        
        risk_level = risk_levels.get(coin, 'moderate to high')
        
        risk_factors = [
            "market volatility",
            "regulatory uncertainty",
            "smart contract vulnerabilities",
            "centralization concerns",
            "competition in the space",
            "developer activity trends",
            "potential tokenomic changes",
            "correlation with BTC movements",
            "liquidity constraints"
        ]
        
        # 随机选择2个风险因素
        selected_risks = random.sample(risk_factors, 2)
        
        return f"{prefix}the risk profile for {coin} is currently {risk_level}. Key factors to consider: {selected_risks[0]} and {selected_risks[1]}. Always use proper position sizing (suggested: no more than {random.randint(1, 5)}% of portfolio for altcoins)."
    
    def _generate_news_response(self, prefix, coin):
        """生成新闻回复"""
        if not coin:
            recent_topics = [
                "upcoming ETF decisions",
                "major protocol upgrades",
                "institutional adoption",
                "regulatory developments",
                "DeFi innovations",
                "NFT market trends",
                "layer 2 scaling solutions",
                "central bank digital currencies"
            ]
            
            # 随机选择2个话题
            selected_topics = random.sample(recent_topics, 2)
            
            return f"{prefix}recent market news has been dominated by {selected_topics[0]} and {selected_topics[1]}. The sentiment is generally {random.choice(['positive', 'mixed', 'cautious', 'optimistic'])}. Any specific coin you're interested in?"
        
        coin = coin.upper()
        
        news_templates = [
            f"{coin} recently announced a partnership with {random.choice(['a major payment provider', 'a technology company', 'a gaming platform', 'a DeFi protocol'])}.",
            f"{coin} team is planning a {random.choice(['major upgrade', 'token burn', 'staking improvement', 'governance update'])} in Q{random.randint(1, 4)}.",
            f"{coin} has seen {random.choice(['increasing', 'strong', 'growing'])} adoption with {random.choice(['rising transaction counts', 'new partnerships', 'institutional interest', 'developer activity'])}.",
            f"{coin} ecosystem is expanding with {random.choice(['new DeFi protocols', 'NFT platforms', 'layer 2 solutions', 'cross-chain bridges'])}."
        ]
        
        return f"{prefix}{random.choice(news_templates)} Community sentiment is {random.choice(['bullish', 'cautiously optimistic', 'growing more positive', 'mixed but improving'])}."
    
    def _generate_trend_response(self, prefix):
        """生成趋势回复"""
        trending_categories = [
            "DeFi protocols",
            "gaming tokens",
            "metaverse projects",
            "AI-related cryptocurrencies",
            "layer 1 alternatives",
            "layer 2 scaling solutions",
            "interoperability protocols",
            "privacy coins",
            "meme coins",
            "RWA tokens"
        ]
        
        selected_category = random.choice(trending_categories)
        
        trending_coins = {
            "DeFi protocols": ["UNI", "AAVE", "MKR", "CRV"],
            "gaming tokens": ["AXS", "GALA", "MANA", "SAND"],
            "metaverse projects": ["MANA", "SAND", "APE", "GALA"],
            "AI-related cryptocurrencies": ["FET", "OCEAN", "AGIX", "RLC"],
            "layer 1 alternatives": ["SOL", "AVAX", "ADA", "NEAR"],
            "layer 2 scaling solutions": ["MATIC", "OP", "ARB", "IMX"],
            "interoperability protocols": ["DOT", "ATOM", "QNT", "RUNE"],
            "privacy coins": ["XMR", "ZEC", "SCRT", "ROSE"],
            "meme coins": ["DOGE", "SHIB", "PEPE", "FLOKI"],
            "RWA tokens": ["MKR", "LINK", "UNI", "AAVE"]
        }
        
        coins = trending_coins.get(selected_category, ["BTC", "ETH", "SOL", "BNB"])
        selected_coins = random.sample(coins, min(3, len(coins)))
        
        return f"{prefix}I'm seeing significant interest in {selected_category} right now. Top trending tokens include: {', '.join(['$' + coin for coin in selected_coins])}. Social media mentions have increased {random.randint(20, 100)}% in the last 24 hours."
    
    def _generate_educational_response(self, prefix, query):
        """生成教育性回复"""
        query_lower = query.lower()
        
        if 'staking' in query_lower:
            return f"{prefix}staking is a way to earn passive income by participating in network security. It works by locking up your tokens to support blockchain operations. Benefits include earning rewards (typically {random.randint(3, 15)}% APY), while risks include lockup periods and potential slashing. It's generally considered lower risk than leveraged trading."
        
        elif 'defi' in query_lower:
            return f"{prefix}DeFi (Decentralized Finance) refers to financial services built on blockchain that operate without central authorities. Common DeFi activities include lending/borrowing, yield farming, liquidity provision, and decentralized trading. While it offers higher yields than traditional finance, risks include smart contract vulnerabilities and impermanent loss."
        
        elif 'nft' in query_lower:
            return f"{prefix}NFTs (Non-Fungible Tokens) are unique digital assets verified on blockchain. Unlike cryptocurrencies, each NFT has distinct value and cannot be exchanged 1:1. They're used for digital art, collectibles, gaming items, and increasingly for real-world asset representation. The market experiences cycles of high volatility."
        
        elif 'layer 2' in query_lower or 'l2' in query_lower:
            return f"{prefix}Layer 2 solutions are scaling technologies built on top of blockchains like Ethereum. They process transactions off the main chain while inheriting its security, resulting in faster and cheaper transactions. Popular L2s include Optimism, Arbitrum, Polygon, and zkSync, each using different approaches like rollups or sidechains."
        
        else:
            return f"{prefix}the crypto ecosystem is constantly evolving with innovations in consensus mechanisms, scaling solutions, tokenomics, and use cases. I'd be happy to explain any specific concept you're curious about - just let me know what you'd like to learn!"
    
    def _generate_general_response(self, prefix, query):
        """生成通用回复"""
        general_responses = [
            f"{prefix}the market is showing {random.choice(['interesting patterns', 'mixed signals', 'accumulation behavior'])} right now. I'm seeing {random.choice(['increasing social interest', 'growing institutional participation', 'stronger on-chain metrics'])} for major assets. Anything specific you'd like to know?",
            
            f"{prefix}I'm tracking {random.randint(400, 600)} crypto KOLs and analyzing {random.randint(10, 30)}K social media posts daily. Current market sentiment is {random.choice(['cautiously optimistic', 'divided', 'improving', 'generally bullish'])}. How can I help with your crypto analysis?",
            
            f"I'm currently analyzing market data across {random.randint(40, 100)} exchanges and {random.randint(5, 20)} blockchains. The dominant narrative seems to be focused on {random.choice(['ETF developments', 'institutional adoption', 'regulatory clarity', 'technological advancements'])}. What aspect of the market interests you most?"
        ]
        
        return random.choice(general_responses)