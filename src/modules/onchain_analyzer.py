import logging
import requests
import json
import os
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/onchain_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OnChainAnalyzer:
    def __init__(self):
        self.cache_dir = "data/onchain_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_onchain_metrics(self, assets):
        """Get on-chain indicator data"""
        result = {}
        
        for asset in assets:
            try:
                # 检查缓存
                cache_file = f"{self.cache_dir}/{asset}_metrics.json"
                now = datetime.now()
                
                if os.path.exists(cache_file):
                    file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                    # 如果缓存不到2小时，使用缓存数据
                    if now - file_time < timedelta(hours=2):
                        with open(cache_file, 'r') as f:
                            result[asset] = json.load(f)
                            continue
                
                # 否则获取新数据
                # 在真实环境中，应该使用链上数据API例如Glassnode、IntoTheBlock等
                
                # 由于API限制，使用模拟数据作为示例
                asset_data = self._get_mock_onchain_data(asset)
                
                # 保存到缓存
                with open(cache_file, 'w') as f:
                    json.dump(asset_data, f, indent=2)
                    
                result[asset] = asset_data
                
            except Exception as e:
                logger.error(f"Error when getting {asset} on-chain metrics: {str(e)}")
                result[asset] = self._get_mock_onchain_data(asset)  # 出错时返回模拟数据
                
        return result
        
    def track_whale_movements(self, threshold_btc=100, threshold_eth=1000):
        """Tracking the movements of big investors"""
        try:
            # 在真实环境中，这需要使用专门的区块链数据API
            # 以下是模拟数据
            return self._get_mock_whale_movements()
            
        except Exception as e:
            logger.error(f"Error in tracking big investors: {str(e)}")
            return []
            
    def _get_mock_onchain_data(self, asset):
        """Generate simulated on-chain data"""
        if asset.lower() == 'bitcoin':
            return {
                'active_addresses': 950000,
                'transaction_count_24h': 285000,
                'transaction_volume': 14500000000,  # 美元
                'average_transaction_value': 52000,  # 美元
                'mempool_size': 15000,
                'hash_rate': 350000000000000000000,  # 350 EH/s
                'difficulty': 55000000000000,
                'whale_count': 2150,  # 持有超过100 BTC的地址数
                'supply_last_active': {
                    '24h': 1.2,  # 百分比
                    '1w': 3.8,
                    '1m': 12.5,
                    '1y': 65.3
                }
            }
        elif asset.lower() == 'ethereum':
            return {
                'active_addresses': 650000,
                'transaction_count_24h': 1150000,
                'transaction_volume': 8700000000,  # 美元
                'average_transaction_value': 7800,  # 美元
                'gas_used_24h': 78000000000,
                'average_gas_price': 25,  # Gwei
                'total_value_locked': 32500000000,  # DeFi中锁定的ETH价值（美元）
                'staked_eth': 25000000,  # ETH数量
                'whale_count': 1420,  # 持有超过1000 ETH的地址数
                'supply_last_active': {
                    '24h': 2.5,  # 百分比
                    '1w': 6.2,
                    '1m': 18.7,
                    '1y': 72.1
                }
            }
        else:
            # 默认数据
            return {
                'active_addresses': 75000,
                'transaction_count_24h': 85000,
                'transaction_volume': 950000000,  # 美元
                'average_transaction_value': 11500  # 美元
            }
            
    def _get_mock_whale_movements(self):
        """Generate simulated large-scale investor movement data"""
        return [
            {
                'blockchain': 'Bitcoin',
                'from_address': '34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo',
                'to_address': 'Exchange (Binance)',
                'amount': 1250,
                'amount_usd': 37500000,
                'timestamp': (datetime.now() - timedelta(hours=3)).isoformat(),
                'transaction_type': 'transfer'
            },
            {
                'blockchain': 'Ethereum',
                'from_address': 'Exchange (Coinbase)',
                'to_address': '0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503',
                'amount': 15000,
                'amount_usd': 22500000,
                'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(),
                'transaction_type': 'transfer'
            },
            {
                'blockchain': 'Bitcoin',
                'from_address': '1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ',
                'to_address': 'Exchange (Kraken)',
                'amount': 850,
                'amount_usd': 25500000,
                'timestamp': (datetime.now() - timedelta(hours=12)).isoformat(),
                'transaction_type': 'transfer'
            }
        ]