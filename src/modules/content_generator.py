# modules/content_generator.py
import os
import json
import random
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class ContentGenerator:
    def __init__(self, openai_client=None, nlp_analyzer=None):
        self.openai_client = openai_client
        self.nlp_analyzer = nlp_analyzer
        self.templates_dir = "data/templates"
        self.templates = self._load_templates()
        
        # 确保目录存在
        os.makedirs(self.templates_dir, exist_ok=True)
        
    def _load_templates(self):
        """加载内容模板"""
        templates = {
            "market_update": [],
            "trend_analysis": [],
            "technical_analysis": [],
            "risk_assessment": [],
            "news_summary": []
        }
        
        # 检查并加载现有模板
        for template_type in templates.keys():
            template_file = f"{self.templates_dir}/{template_type}_templates.json"
            
            if os.path.exists(template_file):
                try:
                    with open(template_file, 'r') as f:
                        loaded_templates = json.load(f)
                        templates[template_type] = loaded_templates
                        logger.info(f"Loaded {len(loaded_templates)} {template_type} templates")
                except Exception as e:
                    logger.error(f"Error loading {template_type} templates: {str(e)}")
            else:
                # 如果模板文件不存在，创建默认模板
                self._create_default_templates(template_type, template_file)
                
        return templates
        
    def _create_default_templates(self, template_type, file_path):
        """创建默认模板"""
        default_templates = {
            "market_update": [
                "🔥 Market Alert: {trend_direction} momentum building in #{main_coin}! {sentiment_description}. Key support at ${support_level}, resistance at ${resistance_level}. #{trending_topic1} #{trending_topic2}",
                
                "📊 {timeframe} Update:\n#{main_coin}: {price_change}% ({sentiment_emoji})\nVolume: ${volume}M\nTop Gainers: #{top_gainer1} {top_gainer1_change}%, #{top_gainer2} {top_gainer2_change}%\nTrending: #{trending_topic1}",
                
                "👀 Watching #{main_coin} closely as it {price_action} ${price_level}. {volume_description}. Community sentiment: {sentiment_label}. Stay tuned for more updates!"
            ],
            "trend_analysis": [
                "🔍 Trend Analysis: {trending_sector} sector showing {momentum_description}. Key players: ${token1}, ${token2}, ${token3}. {reason_for_trend}",
                
                "📈 Sector Rotation: Capital flowing from {declining_sector} into {rising_sector}. This pattern typically {pattern_outcome}. Tokens to watch: ${token1}, ${token2}",
                
                "🌊 Market Cycle Update: We're currently in the {cycle_phase} phase for #{main_coin}. Historical patterns suggest {expected_outcome}. Key indicators: {indicator1}, {indicator2}"
            ],
            "technical_analysis": [
                "⚡ Technical Alert: #{symbol} {pattern_type} pattern formed on the {timeframe} chart. Target: ${target_price}. Stop: ${stop_price}. {confidence_level} confidence",
                
                "📉 #{symbol} Technical Analysis:\nTrend: {trend_direction}\nKey Levels: Support ${support}, Resistance ${resistance}\nIndicators: RSI {rsi_value}, MACD {macd_status}\nOutlook: {outlook}",
                
                "🔑 Key Level Alert: #{symbol} approaching critical {level_type} at ${price_level}. {historical_context}. Watching for {expected_reaction}"
            ],
            "risk_assessment": [
                "⚠️ Risk Assessment: #{token} currently shows {risk_level} risk profile. Metrics: Volatility {volatility_score}/10, Liquidity {liquidity_score}/10, Sentiment {sentiment_score}/10",
                
                "🛡️ Market Risk Update: Overall crypto market risk is {market_risk_level}. Key factors: {risk_factor1}, {risk_factor2}, {risk_factor3}. Suggested approach: {risk_strategy}",
                
                "🔮 #{token} Risk Scorecard:\nTechnical: {technical_risk}/10\nFundamental: {fundamental_risk}/10\nSocial: {social_risk}/10\nOverall: {overall_risk}/10\nVerdict: {risk_verdict}"
            ],
            "news_summary": [
                "📰 Latest Crypto News:\n1. {headline1}\n2. {headline2}\n3. {headline3}\nMarket reaction: {market_reaction}",
                
                "🗞️ {timeframe} News Roundup: {major_headline} dominates discussions. Also trending: {other_headline1}, {other_headline2}. Impact on market: {market_impact}",
                
                "📣 Breaking: {breaking_news_headline}. Early analysis suggests this could {potential_impact}. #{related_token1} #{related_token2}"
            ]
        }
        
        if template_type in default_templates:
            templates = default_templates[template_type]
            
            # 保存模板到文件
            try:
                with open(file_path, 'w') as f:
                    json.dump(templates, f, indent=2)
                    logger.info(f"Created default {template_type} templates")
                    return templates
            except Exception as e:
                logger.error(f"Error creating default {template_type} templates: {str(e)}")
                
        return []
        
    def generate_market_update(self, market_data, social_analysis):
        """生成市场更新推文"""
        try:
            # 如果有OpenAI客户端可用，使用高级生成
            if self.openai_client and hasattr(self.openai_client, 'chat') and self._check_openai_access():
                return self._generate_with_ai(
                    prompt_type="market_update",
                    market_data=market_data,
                    social_analysis=social_analysis
                )
            
            # 否则使用模板系统
            templates = self.templates["market_update"]
            if not templates:
                templates = [
                    "📊 Crypto Market Update: #BTC ${btc_price}, #ETH ${eth_price}. Market sentiment: {sentiment}. Top gainers: #{gainer1}, #{gainer2}."
                ]
            
            template = random.choice(templates)
            
            # 准备模板变量
            template_vars = self._prepare_market_template_variables(market_data, social_analysis)
            
            # 填充模板变量
            filled_template = self._fill_template(template, template_vars)
            
            return filled_template
            
        except Exception as e:
            logger.error(f"Error generating market update: {str(e)}")
            # 简单的后备模板
            return f"📊 Crypto Market Update: BTC ${self._get_safe_price(market_data, 'BTC', 60000)}. Overall sentiment: {self._get_sentiment_label(social_analysis)}. #Crypto #Bitcoin"
    
    def generate_trend_analysis(self, market_data, social_analysis):
        """生成趋势分析推文"""
        try:
            # 如果有OpenAI客户端可用，使用高级生成
            if self.openai_client and hasattr(self.openai_client, 'chat') and self._check_openai_access():
                return self._generate_with_ai(
                    prompt_type="trend_analysis",
                    market_data=market_data,
                    social_analysis=social_analysis
                )
            
            # 否则使用模板系统
            templates = self.templates["trend_analysis"]
            if not templates:
                templates = [
                    "🔍 Trend Analysis: {sector} tokens showing momentum. Projects to watch: #{token1}, #{token2}, #{token3}. Social media mentions up {mention_increase}% in 24h."
                ]
            
            template = random.choice(templates)
            
            # 准备趋势分析的模板变量
            template_vars = self._prepare_trend_template_variables(market_data, social_analysis)
            
            # 填充模板
            filled_template = self._fill_template(template, template_vars)
            
            return filled_template
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {str(e)}")
            trending_tokens = ["BTC", "ETH", "SOL", "AVAX", "LINK"]
            tokens = random.sample(trending_tokens, 3)
            return f"🔍 Trend Watch: Notable activity in {', '.join(['$' + token for token in tokens])}. Based on social media and market indicators. #Crypto"
    
    def generate_technical_analysis(self, symbol, market_data):
        """为特定币种生成技术分析推文"""
        try:
            # 如果有OpenAI客户端可用，使用高级生成
            if self.openai_client and hasattr(self.openai_client, 'chat') and self._check_openai_access():
                return self._generate_with_ai(
                    prompt_type="technical_analysis",
                    symbol=symbol,
                    market_data=market_data
                )
            
            # 否则使用模板系统
            templates = self.templates["technical_analysis"]
            if not templates:
                templates = [
                    "📉 #{symbol} Technical Analysis:\nTrend: {trend_direction}\nSupport: ${support}\nResistance: ${resistance}\nRSI: {rsi_value}\nOutlook: {outlook}"
                ]
            
            template = random.choice(templates)
            
            # 准备技术分析的模板变量
            template_vars = self._prepare_technical_template_variables(symbol, market_data)
            
            # 填充模板
            filled_template = self._fill_template(template, template_vars)
            
            return filled_template
            
        except Exception as e:
            logger.error(f"Error generating technical analysis: {str(e)}")
            price = self._get_safe_price(market_data, symbol, 1000)
            return f"📊 ${symbol} Technical Analysis: Currently trading at ${price}. Watch key support and resistance levels. Always DYOR. #{symbol} #Crypto"
    
    def _check_openai_access(self):
        """检查OpenAI API是否可用"""
        try:
            # 简单的测试请求
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello, just a quick test."}
                ],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI API access check failed: {str(e)}")
            return False
            
    def _generate_with_ai(self, prompt_type, **kwargs):
        """使用OpenAI API生成内容"""
        try:
            # 准备提示
            if prompt_type == "market_update":
                prompt = self._prepare_market_update_prompt(kwargs.get('market_data', {}), kwargs.get('social_analysis', {}))
            elif prompt_type == "trend_analysis":
                prompt = self._prepare_trend_analysis_prompt(kwargs.get('market_data', {}), kwargs.get('social_analysis', {}))
            elif prompt_type == "technical_analysis":
                prompt = self._prepare_technical_analysis_prompt(kwargs.get('symbol', 'BTC'), kwargs.get('market_data', {}))
            else:
                prompt = f"Write a short, informative tweet about the current cryptocurrency market conditions."
            
            # 调用API
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a highly knowledgeable cryptocurrency market analyst who writes concise, insightful tweets with relevant hashtags. Your tone is professional yet approachable, and you favor data-driven insights over speculation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=280,  # Twitter-friendly length
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # 确保文本长度符合Twitter限制
            if len(generated_text) > 280:
                generated_text = generated_text[:277] + "..."
                
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating content with AI: {str(e)}")
            return None
    
    def _prepare_market_update_prompt(self, market_data, social_analysis):
        """准备市场更新的AI提示"""
        # 提取市场数据
        btc_price = self._get_safe_price(market_data, 'BTC', 65000)
        eth_price = self._get_safe_price(market_data, 'ETH', 3500)
        market_cap = market_data.get('total_market_cap', '$2.5T')
        
        # 提取社交分析数据
        sentiment = self._get_sentiment_label(social_analysis)
        trending_topics = social_analysis.get('trending_topics', ['bitcoin', 'ethereum', 'defi'])[:3]
        
        # 构建提示
        prompt = f"""
        Create a concise cryptocurrency market update tweet based on the following data:

        Market Data:
        - BTC Price: ${btc_price}
        - ETH Price: ${eth_price}
        - Total Market Cap: {market_cap}

        Social Sentiment: {sentiment}

        Trending Topics: {', '.join(trending_topics)}

        Make it informative yet brief (under 280 characters), include relevant hashtags, and focus on the most important insights. Use crypto-native language and include emoji where appropriate.
        """
        
        return prompt
    
    def _prepare_trend_analysis_prompt(self, market_data, social_analysis):
        """准备趋势分析的AI提示"""
        # 提取趋势数据
        trending_topics = social_analysis.get('trending_topics', ['DeFi', 'NFTs', 'Layer2'])[:5]
        trending_tokens = social_analysis.get('mentioned_tokens', ['BTC', 'ETH', 'SOL', 'AVAX', 'ARB'])[:5]
        
        # 构建提示
        prompt = f"""
        Create an insightful tweet analyzing current cryptocurrency trends based on the following data:

        Trending Topics: {', '.join(trending_topics)}
        
        Frequently Mentioned Tokens: {', '.join(trending_tokens)}
        
        Overall Market Sentiment: {self._get_sentiment_label(social_analysis)}

        Identify the most significant trend, explain why it matters, and mention 2-3 relevant tokens. 
        Keep it under 280 characters, include relevant hashtags, and use crypto-native language with appropriate emoji.
        """
        
        return prompt
    
    def _prepare_technical_analysis_prompt(self, symbol, market_data):
        """准备技术分析的AI提示"""
        # 尝试获取技术指标
        try:
            technical_data = market_data.get('technical_indicators', {}).get(symbol, {})
        except:
            technical_data = {}
            
        # 如果没有技术数据，创建一些模拟数据
        price = self._get_safe_price(market_data, symbol, 1000)
        rsi = technical_data.get('rsi', random.randint(30, 70))
        ma_status = technical_data.get('ma_status', random.choice(['bullish', 'bearish', 'neutral']))
        volume_change = technical_data.get('volume_change', random.randint(-30, 30))
        
        # 构建提示
        prompt = f"""
        Create a concise technical analysis tweet for ${symbol} based on the following indicators:

        Current Price: ${price}
        RSI: {rsi}
        Moving Average Status: {ma_status}
        24h Volume Change: {volume_change}%

        Identify key support/resistance levels, mention relevant patterns if any, and give a short-term outlook.
        Keep it under 280 characters, include relevant hashtags (#{symbol}, #Crypto, etc.), and use appropriate technical analysis terminology and emoji.
        """
        
        return prompt
    
    def _prepare_market_template_variables(self, market_data, social_analysis):
        """准备市场更新的模板变量"""
        # 获取基本价格数据
        btc_price = self._get_safe_price(market_data, 'BTC', 65000)
        eth_price = self._get_safe_price(market_data, 'ETH', 3500)
        
        # 分析社交数据
        sentiment = self._get_sentiment_label(social_analysis)
        sentiment_emoji = "🟢" if sentiment == "bullish" else "🔴" if sentiment == "bearish" else "⚪"
        
        # 生成有逻辑的变量值
        trend_direction = random.choice(["bullish", "bearish", "neutral"])
        if trend_direction == "bullish":
            sentiment_description = random.choice(["Buyers taking control", "Bulls pushing higher", "Strong demand"])
            price_action = random.choice(["breaks above", "tests resistance at", "clears"])
        elif trend_direction == "bearish":
            sentiment_description = random.choice(["Sellers stepping in", "Bears active", "Profit taking phase"])
            price_action = random.choice(["drops below", "tests support at", "struggles at"])
        else:
            sentiment_description = random.choice(["Consolidation phase", "Sideways action", "Low volatility"])
            price_action = random.choice(["ranges near", "consolidates around", "trades at"])
        
        # 准备一组有代表性的模板变量
        return {
            "btc_price": btc_price,
            "eth_price": eth_price,
            "main_coin": "BTC",
            "timeframe": random.choice(["Daily", "4h", "Weekly"]),
            "price_change": random.uniform(-5.0, 5.0),
            "trend_direction": trend_direction,
            "sentiment_description": sentiment_description,
            "sentiment_label": sentiment,
            "sentiment_emoji": sentiment_emoji,
            "price_action": price_action,
            "price_level": btc_price,
            "support_level": round(btc_price * 0.93, -3),  # 向下取整到千位
            "resistance_level": round(btc_price * 1.07, -3),  # 向上取整到千位
            "volume": round(random.uniform(20000, 50000), -3),  # 四舍五入到千位
            "volume_description": random.choice(["Volume increasing", "Volume declining", "Average volume"]),
            "top_gainer1": random.choice(["SOL", "AVAX", "LINK", "DOT"]),
            "top_gainer1_change": random.uniform(5.0, 15.0),
            "top_gainer2": random.choice(["ARB", "OP", "MATIC", "ADA"]),
            "top_gainer2_change": random.uniform(3.0, 10.0),
            "trending_topic1": random.choice(["Bitcoin", "Crypto", "Blockchain", "DeFi", "NFT"]),
            "trending_topic2": random.choice(["Altseason", "Trading", "ETF", "Web3", "Metaverse"])
        }
    
    def _prepare_trend_template_variables(self, market_data, social_analysis):
        """准备趋势分析的模板变量"""
        # 分析趋势领域
        sectors = ["DeFi", "NFT", "Gaming", "Layer1", "Layer2", "Meme", "AI", "RWA", "Privacy", "Metaverse"]
        sector_pairs = list(zip(sectors, random.sample(sectors, len(sectors))))
        
        trending_sector = random.choice(sectors)
        declining_sector = next((s[1] for s in sector_pairs if s[0] == trending_sector), random.choice(sectors))
        
        # 准备代币列表
        sector_tokens = {
            "DeFi": ["UNI", "AAVE", "MKR", "CRV"],
            "NFT": ["APE", "BLUR", "GALA", "IMX"],
            "Gaming": ["AXS", "GALA", "SAND", "ENJ"],
            "Layer1": ["SOL", "ADA", "AVAX", "NEAR"],
            "Layer2": ["MATIC", "OP", "ARB", "IMX"],
            "Meme": ["DOGE", "SHIB", "PEPE", "FLOKI"],
            "AI": ["FET", "OCEAN", "AGIX", "RLC"],
            "RWA": ["MKR", "LINK", "UNI", "AAVE"],
            "Privacy": ["XMR", "ZEC", "SCRT", "ROSE"],
            "Metaverse": ["MANA", "SAND", "AXS", "ENJ"]
        }
        
        # 获取相关代币
        tokens = sector_tokens.get(trending_sector, ["BTC", "ETH", "SOL"])
        tokens = random.sample(tokens, min(3, len(tokens)))
        
        # 准备变量
        return {
            "trending_sector": trending_sector,
            "declining_sector": declining_sector,
            "token1": tokens[0] if len(tokens) > 0 else "BTC",
            "token2": tokens[1] if len(tokens) > 1 else "ETH",
            "token3": tokens[2] if len(tokens) > 2 else "SOL",
            "momentum_description": random.choice(["growing momentum", "increasing interest", "bullish sentiment", "rising social mentions"]),
            "reason_for_trend": random.choice([
                "Driven by recent protocol upgrades",
                "Following major partnership announcements",
                "As institutional interest grows",
                "After strong on-chain activity metrics"
            ]),
            "pattern_outcome": random.choice([
                "continues for 1-2 weeks before reversing",
                "signals a longer-term market rotation",
                "presents short-term trading opportunities",
                "often precedes major market moves"
            ]),
            "cycle_phase": random.choice(["accumulation", "mark up", "distribution", "mark down", "early recovery"]),
            "expected_outcome": random.choice([
                "continued upward momentum for 2-3 weeks",
                "a period of consolidation before the next move",
                "increased volatility as traders position themselves",
                "clearer directional movement after current uncertainty"
            ]),
            "indicator1": random.choice(["market structure", "volume profile", "funding rates", "derivatives data"]),
            "indicator2": random.choice(["social sentiment", "exchange flows", "holder distribution", "institutional positioning"])
        }
    
    def _prepare_technical_template_variables(self, symbol, market_data):
        """准备技术分析的模板变量"""
        # 获取基本价格
        price = self._get_safe_price(market_data, symbol, 1000)
        
        # 生成随机但合理的技术指标
        trend_choices = ["bullish", "bearish", "neutral", "consolidating", "ranging"]
        trend_probabilities = [0.4, 0.3, 0.15, 0.1, 0.05]  # 权重偏向明确方向
        trend_direction = random.choices(trend_choices, weights=trend_probabilities, k=1)[0]
        
        # 根据趋势生成支撑/阻力
        if trend_direction == "bullish":
            support = round(price * random.uniform(0.92, 0.98), 1)
            resistance = round(price * random.uniform(1.05, 1.15), 1)
            rsi_value = random.randint(55, 75)
            macd_status = random.choice(["bullish crossover", "above signal line", "positive and rising"])
            outlook = random.choice(["bullish", "positive with upside potential", "likely to test resistance"])
        elif trend_direction == "bearish":
            support = round(price * random.uniform(0.85, 0.95), 1)
            resistance = round(price * random.uniform(1.01, 1.08), 1)
            rsi_value = random.randint(25, 45)
            macd_status = random.choice(["bearish crossover", "below signal line", "negative and falling"])
            outlook = random.choice(["bearish", "negative with downside risk", "likely to test support"])
        else:
            support = round(price * random.uniform(0.94, 0.98), 1)
            resistance = round(price * random.uniform(1.02, 1.06), 1)
            rsi_value = random.randint(40, 60)
            macd_status = random.choice(["neutral", "near signal line", "flat"])
            outlook = random.choice(["neutral", "range-bound", "awaiting clearer direction"])
        
        # 准备所有模板变量
        return {
            "symbol": symbol,
            "trend_direction": trend_direction,
            "support": support,
            "resistance": resistance,
            "rsi_value": rsi_value,
            "macd_status": macd_status,
            "outlook": outlook,
            "pattern_type": random.choice(["bullish flag", "head and shoulders", "double bottom", "triangle", "channel"]),
            "timeframe": random.choice(["4h", "daily", "weekly"]),
            "target_price": round(price * (1.1 if trend_direction == "bullish" else 0.9), 1),
            "stop_price": round(price * (0.95 if trend_direction == "bullish" else 1.05), 1),
            "confidence_level": random.choice(["High", "Medium", "Moderate"]),
            "level_type": "support" if trend_direction == "bearish" else "resistance",
            "price_level": resistance if trend_direction == "bullish" else support,
            "historical_context": random.choice([
                "This level has been tested 3 times before",
                "Previous reaction at this level was significant",
                "Key level from market structure",
                "Confluence with major moving average"
            ]),
            "expected_reaction": random.choice([
                "potential breakout with volume",
                "possible rejection and reversal",
                "increased volatility near this level",
                "confirmation of trend direction"
            ])
        }
        
    def _get_safe_price(self, market_data, symbol, default_price=None):
        """安全地获取价格，避免KeyError"""
        try:
            # 尝试不同的数据结构格式
            if isinstance(market_data, dict):
                # 直接访问
                if symbol in market_data:
                    price = market_data[symbol]
                # 检查prices子字典
                elif 'prices' in market_data and symbol in market_data['prices']:
                    price = market_data['prices'][symbol]
                # 检查具体的价格字段
                elif f"{symbol.lower()}_price" in market_data:
                    price = market_data[f"{symbol.lower()}_price"]
                elif f"{symbol}_price" in market_data:
                    price = market_data[f"{symbol}_price"]
                else:
                    # 检查嵌套结构
                    for key, value in market_data.items():
                        if isinstance(value, dict) and symbol in value:
                            if 'price' in value[symbol]:
                                price = value[symbol]['price']
                                break
                    else:
                        # 没找到，使用默认值
                        return default_price or (10 if symbol != "BTC" else 60000)
            else:
                return default_price or (10 if symbol != "BTC" else 60000)
                
            # 可能的数据清理
            if isinstance(price, str):
                # 移除可能的货币符号和格式化字符
                price = re.sub(r'[^\d.]', '', price)
                price = float(price)
                
            return price
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return default_price or (10 if symbol != "BTC" else 60000)
    
    def _get_sentiment_label(self, social_analysis):
        """从社交分析中获取情绪标签"""
        try:
            if isinstance(social_analysis, dict):
                # 直接访问sentiment字段
                if 'sentiment' in social_analysis:
                    sentiment_data = social_analysis['sentiment']
                    
                    # 检查不同可能的数据结构
                    if isinstance(sentiment_data, dict) and 'sentiment_label' in sentiment_data:
                        return sentiment_data['sentiment_label']
                    elif isinstance(sentiment_data, dict) and 'label' in sentiment_data:
                        return sentiment_data['label']
                    elif isinstance(sentiment_data, (int, float)):
                        # 数值情绪转换为标签
                        if sentiment_data > 0.1:
                            return "bullish"
                        elif sentiment_data < -0.1:
                            return "bearish"
                        else:
                            return "neutral"
                    elif isinstance(sentiment_data, str):
                        return sentiment_data
                
                # 检查情绪字段的其他可能位置
                for key in ['market_sentiment', 'overall_sentiment', 'community_sentiment']:
                    if key in social_analysis:
                        sentiment = social_analysis[key]
                        if isinstance(sentiment, str):
                            return sentiment
                
            # 未找到情绪数据，随机生成一个
            return random.choice(["bullish", "neutral", "bearish"])
            
        except Exception as e:
            logger.error(f"Error extracting sentiment label: {str(e)}")
            return "neutral"
    
    def _fill_template(self, template, variables):
        """填充模板变量"""
        try:
            # 使用Python的格式化字符串功能
            filled_template = template.format(**variables)
            
            # 检查是否所有占位符都已替换
            if '{' in filled_template and '}' in filled_template:
                # 找出未替换的变量
                unresolved_vars = re.findall(r'\{([^}]+)\}', filled_template)
                
                # 为未替换的变量生成随机替代品
                for var in unresolved_vars:
                    # 根据变量名生成合理的替代值
                    # 根据变量名生成合理的替代值
                    if 'price' in var.lower():
                        replacement = str(random.randint(10, 100000))
                    elif 'percentage' in var.lower() or 'change' in var.lower():
                        replacement = f"{random.uniform(-10, 10):.1f}%"
                    elif 'token' in var.lower() or 'coin' in var.lower():
                        replacement = random.choice(["BTC", "ETH", "SOL", "AVAX", "MATIC"])
                    elif 'topic' in var.lower():
                        replacement = random.choice(["Bitcoin", "Crypto", "Blockchain", "DeFi", "NFT"])
                    else:
                        replacement = "data"
                        
                    # 替换变量
                    filled_template = filled_template.replace(f"{{{var}}}", replacement)
            
            return filled_template
            
        except KeyError as e:
            logger.error(f"Missing template variable: {str(e)}")
            # 返回带有一些基本替换的模板
            return template.replace("{main_coin}", "BTC").replace("{sentiment_label}", "neutral").replace("{trending_topic1}", "Crypto").replace("{trending_topic2}", "Bitcoin")
            
        except Exception as e:
            logger.error(f"Error filling template: {str(e)}")
            return f"📊 Crypto Market Update: BTC ${self._get_safe_price(None, 'BTC', 60000)}. #Crypto #Bitcoin"