import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/anomaly_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self):
        self.anomaly_threshold = 2.5  # Standard deviation threshold
        self.anomaly_history = {}
        
    def detect_market_anomalies(self, top_coins, whale_txs=None):
        """Detect market anomalies"""
        anomalies = []
        
        try:
            # Detect price anomalies
            price_anomalies = self._detect_price_anomalies(top_coins)
            if price_anomalies:
                anomalies.extend(price_anomalies)
            
            # Detect volume anomalies
            volume_anomalies = self._detect_volume_anomalies(top_coins)
            if volume_anomalies:
                anomalies.extend(volume_anomalies)
            
            # Detect whale activity anomalies
            if whale_txs:
                whale_anomalies = self._detect_whale_anomalies(whale_txs)
                if whale_anomalies:
                    anomalies.extend(whale_anomalies)
                    
            # Record anomalies
            self._record_anomalies(anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting market anomalies: {str(e)}")
            return []
            
    def _detect_price_anomalies(self, top_coins):
        """Detect abnormal price movements"""
        anomalies = []
        
        try:
            # Create DataFrame
            df = pd.DataFrame(top_coins)
            
            # Check if required columns exist
            if 'price_change_percentage_24h' not in df.columns or 'symbol' not in df.columns:
                return []
                
            # Calculate mean and standard deviation of price changes
            mean_change = df['price_change_percentage_24h'].mean()
            std_change = df['price_change_percentage_24h'].std()
            
            # If standard deviation is too small, might be a data issue, use default
            if std_change < 0.5:
                std_change = 3.0  # Default standard deviation
                
            # Identify outliers
            for _, coin in df.iterrows():
                change = coin['price_change_percentage_24h']
                
                # If change exceeds threshold
                if abs(change - mean_change) > self.anomaly_threshold * std_change:
                    direction = "surge" if change > 0 else "drop"
                    confidence = min(100, abs((change - mean_change) / std_change) * 20)
                    
                    anomalies.append({
                        'asset': coin['symbol'].upper(),
                        'type': 'price',
                        'description': f"abnormal {direction} of {abs(change):.1f}% in 24h",
                        'value': change,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': confidence
                    })
                    
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting price anomalies: {str(e)}")
            return []
            
    def _detect_volume_anomalies(self, top_coins):
        """Detect abnormal volume increases"""
        anomalies = []
        
        try:
            # Create DataFrame
            df = pd.DataFrame(top_coins)
            
            # Check if required columns exist
            if 'total_volume' not in df.columns or 'symbol' not in df.columns:
                return []
                
            # Calculate mean and standard deviation of volume
            mean_volume = df['total_volume'].mean()
            std_volume = df['total_volume'].std()
            
            # Identify outliers
            for _, coin in df.iterrows():
                volume = coin['total_volume']
                
                # If volume is abnormally high
                if volume > mean_volume + self.anomaly_threshold * std_volume:
                    # Calculate confidence score
                    confidence = min(95, ((volume - mean_volume) / std_volume) * 15)
                    
                    anomalies.append({
                        'asset': coin['symbol'].upper(),
                        'type': 'volume',
                        'description': f"abnormal volume increase (${volume/1000000:.1f}M)",
                        'value': volume,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': confidence
                    })
                    
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting volume anomalies: {str(e)}")
            return []
            
    def _detect_whale_anomalies(self, whale_txs):
        """Detect abnormal whale activity"""
        anomalies = []
        
        try:
            # If multiple large transactions to the same target address
            target_addresses = {}
            
            for tx in whale_txs:
                target = tx.get('to_address', '')
                if 'Exchange' in target:  # If transaction to exchange
                    if target in target_addresses:
                        target_addresses[target].append(tx)
                    else:
                        target_addresses[target] = [tx]
            
            # Check for multiple large transfers to the same exchange
            for address, txs in target_addresses.items():
                if len(txs) >= 2:  # Multiple transactions to the same target
                    total_value = sum(tx.get('amount_usd', 0) for tx in txs)
                    blockchain = txs[0].get('blockchain', 'Unknown')
                    
                    if total_value > 10000000:  # Total value over $10 million
                        anomalies.append({
                            'asset': blockchain,
                            'type': 'whale',
                            'description': f"multiple large transfers to {address} (total ${total_value/1000000:.1f}M)",
                            'value': total_value,
                            'timestamp': datetime.now().isoformat(),
                            'confidence': 85
                        })
                        
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting whale anomalies: {str(e)}")
            return []
            
    def _record_anomalies(self, anomalies):
        """Record anomalies for historical tracking"""
        try:
            for anomaly in anomalies:
                asset = anomaly['asset']
                anomaly_type = anomaly['type']
                key = f"{asset}_{anomaly_type}"
                
                if key not in self.anomaly_history:
                    self.anomaly_history[key] = []
                    
                # Record simplified version of anomaly
                self.anomaly_history[key].append({
                    'timestamp': anomaly['timestamp'],
                    'value': anomaly['value'],
                    'confidence': anomaly['confidence']
                })
                
                # Only keep the most recent 10 records
                if len(self.anomaly_history[key]) > 10:
                    self.anomaly_history[key] = self.anomaly_history[key][-10:]
                    
        except Exception as e:
            logger.error(f"Error recording anomaly history: {str(e)}")