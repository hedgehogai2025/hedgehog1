# src/main.py
import os
import json
import time
import logging
import schedule
from datetime import datetime, timedelta

# 修改导入方式以匹配现有结构
from modules.market_data import MarketData
from modules.nlp_analyzer import NLPAnalyzer
from modules.twitter_client import TwitterClient
from modules.technical_analysis import TechnicalAnalysis
from modules.chart_generator import ChartGenerator

# 导入新模块
from modules.social_data_collector import SocialDataCollector
from modules.user_interaction import UserInteractionHandler
from modules.content_generator import ContentGenerator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保必要的目录存在
os.makedirs('data/charts', exist_ok=True)
os.makedirs('data/signals', exist_ok=True)
os.makedirs('data/social', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def main():
    """主程序入口"""
    logger.info("Starting Cryptocurrency Analysis Bot...")
    
    # 初始化各个模块
    twitter_client = TwitterClient()
    nlp_analyzer = NLPAnalyzer()
    tech_analyzer = TechnicalAnalysis()
    chart_gen = ChartGenerator()
    market_data_client = MarketData()  # 使用您的MarketData类
    
    # 初始化新模块
    social_collector = SocialDataCollector(twitter_client)
    user_handler = UserInteractionHandler(twitter_client, nlp_analyzer, market_data_client)
    content_generator = ContentGenerator(openai_client=None, nlp_analyzer=nlp_analyzer)
    
    # 定义任务
    def collect_expanded_data():
        """收集扩展的市场和社交数据"""
        logger.info("Collecting expanded social and market data...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # 收集市场数据 - 使用MarketData类的方法
        try:
            # 获取顶级加密货币数据
            top_coins_df = market_data_client.get_top_coins(limit=100)
            
            # 计算市场指标
            market_indicators = market_data_client.calculate_market_indicators(top_coins_df)
            
            # 获取趋势币种
            trending_coins = market_data_client.get_trending_coins()
            
            # 构建完整的市场数据对象
            market_data_result = {
                'btc_price': float(top_coins_df[top_coins_df['id'] == 'bitcoin']['current_price'].values[0]) if 'bitcoin' in top_coins_df['id'].values else 65000,
                'eth_price': float(top_coins_df[top_coins_df['id'] == 'ethereum']['current_price'].values[0]) if 'ethereum' in top_coins_df['id'].values else 3500,
                'total_market_cap': market_indicators.get('total_market_cap', '$2.5T'),
                'btc_dominance': market_indicators.get('bitcoin_dominance', 48),
                'total_volume_24h': top_coins_df['total_volume'].sum() if not top_coins_df.empty else 100000000000,
                'market_indicators': market_indicators,
                'top_coins': top_coins_df.head(20).to_dict('records') if not top_coins_df.empty else [],
                'trending_coins': trending_coins,
                'timestamp': datetime.now().isoformat()
            }
            
            # 添加交易信号和异常
            market_data_result['signals'] = {}
            market_data_result['anomalies'] = []
            
            # 提取顶级赢家和输家作为趋势和异常
            if 'top_gainers' in market_indicators and market_indicators['top_gainers']:
                for gainer in market_indicators['top_gainers'][:2]:
                    market_data_result['signals'][gainer['symbol'].upper()] = {
                        'direction': 'buy',
                        'strength': min(abs(gainer['price_change_percentage_24h']) / 10, 1.0),
                        'timestamp': datetime.now().isoformat()
                    }
            
            if 'top_losers' in market_indicators and market_indicators['top_losers']:
                for loser in market_indicators['top_losers'][:2]:
                    # 超过10%的损失添加为异常
                    if abs(loser['price_change_percentage_24h']) > 10:
                        market_data_result['anomalies'].append({
                            'asset': loser['symbol'].upper(),
                            'description': f"abnormal drop of {abs(loser['price_change_percentage_24h']):.1f}%",
                            'confidence': min(abs(loser['price_change_percentage_24h']) * 2, 90),
                            'timestamp': datetime.now().isoformat()
                        })
                    # 否则添加为卖出信号
                    else:
                        market_data_result['signals'][loser['symbol'].upper()] = {
                            'direction': 'sell',
                            'strength': min(abs(loser['price_change_percentage_24h']) / 10, 1.0),
                            'timestamp': datetime.now().isoformat()
                        }
        except Exception as e:
            logger.error(f"Error collecting market data: {str(e)}")
            market_data_result = {
                'btc_price': 65000,
                'eth_price': 3500,
                'timestamp': datetime.now().isoformat()
            }
        
        with open(f"data/market_data_{timestamp}.json", 'w') as f:
            json.dump(market_data_result, f)
        logger.info(f"Market data collected and saved to data/market_data_{timestamp}.json")
        
        # 收集社交数据
        reddit_data = nlp_analyzer.fetch_reddit_sentiment()
        kol_tweets = social_collector.collect_kol_tweets(hours=6)  # 获取最近6小时的KOL推文
        community_tweets = social_collector.collect_community_sentiment()
        
        # 合并所有社交数据
        all_social_data = {
            'reddit': reddit_data,
            'kol_tweets': kol_tweets,
            'community': community_tweets
        }
        
        # 分析社交数据
        try:
            if hasattr(nlp_analyzer, 'analyze_alternative_sources'):
                social_analysis = nlp_analyzer.analyze_alternative_sources()
            else:
                # 使用默认的社交分析方法
                social_analysis = {'sentiment': 0.2, 'topics': [('bitcoin', 5), ('ethereum', 4), ('defi', 3)]}
                
                # 准备趋势币种
                trending_tokens = []
                if trending_coins:
                    trending_tokens = [coin['symbol'].upper() for coin in trending_coins[:5]]
                
                # 添加到社交分析结果
                social_analysis['mentioned_tokens'] = trending_tokens
        except Exception as e:
            logger.error(f"Error analyzing social data: {str(e)}")
            social_analysis = {'sentiment': 'neutral'}
        
        # 保存社交数据和分析结果
        social_data_file = f"data/social_data_{timestamp}.json"
        with open(social_data_file, 'w') as f:
            json.dump({
                'data': all_social_data,
                'analysis': social_analysis
            }, f)
        logger.info(f"Social data collected and saved to {social_data_file}")
        
        return market_data_result, social_analysis
    
    def generate_and_post_market_update():
        """生成并发布市场更新"""
        logger.info("Generating market updates...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # 收集数据
        market_data_result, social_analysis = collect_expanded_data()
        
        # 生成市场概览图表
        chart_path = f"data/charts/market_overview_{timestamp}.png"
        try:
            # 尝试使用chart_gen对象
            chart_gen.generate_market_overview(market_data_result, output_path=chart_path)
            logger.info(f"Market overview chart saved to {chart_path}")
        except Exception as e:
            logger.error(f"Error generating market overview chart: {str(e)}")
            chart_path = None
        
        # 生成技术分析图表
        chart_paths = {}
        for symbol in ['BTC', 'ETH', 'BNB', 'SOL', 'XRP']:
            ta_chart_path = f"data/charts/{symbol}_analysis_{timestamp}.png"
            try:
                chart_gen.generate_technical_chart(symbol, output_path=ta_chart_path)
                chart_paths[symbol] = ta_chart_path
                logger.info(f"{symbol} technical analysis chart saved to {ta_chart_path}")
            except Exception as e:
                logger.error(f"Error generating {symbol} technical chart: {str(e)}")
        
        # 生成市场分析文本
        market_update = content_generator.generate_market_update(market_data_result, social_analysis)
        trend_analysis = content_generator.generate_trend_analysis(market_data_result, social_analysis)
        
        # 保存分析到文件
        analysis_file = f"data/analysis_{timestamp}.txt"
        with open(analysis_file, 'w') as f:
            f.write(f"{market_update}\n\n{trend_analysis}")
        logger.info(f"Market analysis saved to {analysis_file}")
        
        # 发布到Twitter
        logger.info("Posting market analysis to Twitter...")
        
        # 发布市场更新
        tweet_id = None
        if chart_path and os.path.exists(chart_path):
            tweet_id = twitter_client.post_tweet_with_media(
                text=market_update,
                media_path=chart_path
            )
        else:
            # 如果没有图表，只发布文本
            tweet_id = twitter_client.post_tweet(market_update)
        
        # 如果有趋势分析，作为回复发布
        if trend_analysis and tweet_id:
            twitter_client.reply_to_tweet(
                tweet_id=tweet_id,
                text=trend_analysis
            )
            logger.info("Analysis thread successfully posted to Twitter")
            
        # 发布交易信号
        if 'signals' in market_data_result and market_data_result['signals']:
            signals_text = "📈 #TradingSignal Alert:\n"
            for symbol, signal in list(market_data_result['signals'].items())[:3]:  # 限制为前3个信号
                direction = "🟢 BUY" if signal.get('direction') == 'buy' else "🔴 SELL"
                strength = signal.get('strength', 0.5)
                signals_text += f"{direction} {symbol}: Strength {strength:.1f}/1.0\n"
                
            twitter_client.post_tweet(signals_text)
            logger.info("Trading signals posted to Twitter")
            
        # 发布异常警报
        if 'anomalies' in market_data_result and market_data_result['anomalies']:
            anomaly_text = "🚨 #MarketAnomaly Alert:\n"
            for anomaly in market_data_result['anomalies'][:2]:  # 限制为前2个异常
                asset = anomaly.get('asset', 'Unknown')
                description = anomaly.get('description', 'unusual activity')
                confidence = anomaly.get('confidence', 50)
                anomaly_text += f"{asset}: {description} ({confidence:.1f}% confidence)\n"
                
            twitter_client.post_tweet(anomaly_text)
            logger.info("Anomaly alerts posted to Twitter")
            
        # 发布简单的市场更新
        btc_price = market_data_result.get('btc_price', '?')
        market_cap = market_data_result.get('total_market_cap', '?')
        volume_24h = market_data_result.get('total_volume_24h', '?')
        btc_dominance = market_data_result.get('btc_dominance', '?')
        
        summary = f"📊 #Crypto Market Update ({datetime.now().strftime('%Y-%m-%d %H:%M')}):\n"
        summary += f"BTC: ${btc_price}\n"
        summary += f"Total Market Cap: ${market_cap:,.0f}\n" if isinstance(market_cap, (int, float)) else f"Total Market Cap: {market_cap}\n"
        summary += f"24h Volume: ${volume_24h:,.0f}\n" if isinstance(volume_24h, (int, float)) else f"24h Volume: {volume_24h}\n"
        summary += f"BTC Dominance: {btc_dominance:.1f}%" if isinstance(btc_dominance, (int, float)) else f"BTC Dominance: {btc_dominance}"
        
        if chart_path and os.path.exists(chart_path):
            twitter_client.post_tweet_with_media(
                text=summary,
                media_path=chart_path
            )
        else:
            twitter_client.post_tweet(summary)
            
        logger.info("Simple market update posted to Twitter")
        logger.info("Market analysis successfully generated")
        
        return True
    
    def check_user_interactions():
        """检查用户互动并回复"""
        logger.info("Checking for user interactions...")
        user_handler.check_and_respond_mentions()
    
    # 设置定时任务
    schedule.every(30).minutes.do(collect_expanded_data)
    schedule.every(1).hours.do(generate_and_post_market_update)
    schedule.every(5).minutes.do(check_user_interactions)
    
    # 主循环
    logger.info("All tasks scheduled, starting main loop")
    
    # 启动时先运行一次数据收集和市场更新
    try:
        generate_and_post_market_update()
    except Exception as e:
        logger.error(f"Error during initial market update: {str(e)}")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次定时任务
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(300)  # 如果出错，等待5分钟再继续

if __name__ == "__main__":
    main()