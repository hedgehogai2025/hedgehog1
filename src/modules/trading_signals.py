import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_signals.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSignals:
    def __init__(self):
        self.signal_dir = "data/signals"
        os.makedirs(self.signal_dir, exist_ok=True)
        
        # Signal thresholds
        self.strong_trend_threshold = 0.7  # Minimum trend strength for a signal
        self.signal_history = {}  # Track signal history
        
    def generate_signals(self, market_data, technical_indicators):
        """Generate trading signals based on technical analysis and market data"""
        signals = {}
        
        try:
            # For each asset with technical indicators
            for asset, indicators in technical_indicators.items():
                if not indicators or len(indicators) == 0:
                    continue
                    
                # Get trend data
                trend_direction = indicators.get('trend_direction', 0)
                trend_strength = indicators.get('trend_strength', 0)
                
                # Skip weak trends
                if trend_strength < 0.5:  # Lowered threshold for more signals
                    continue
                    
                # Determine signal direction
                signal_direction = "buy" if trend_direction > 0 else "sell"
                
                # Get price data
                current_price = indicators.get('last_price', 0)
                if current_price == 0:
                    continue
                    
                # Calculate entry price (current price)
                entry_price = current_price
                
                # Calculate stop loss
                support = indicators.get('support_level', 0)
                resistance = indicators.get('resistance_level', 0)
                
                stop_loss = None
                take_profit = None
                
                if signal_direction == "buy" and support > 0:
                    # For buy signals, set stop loss below support
                    stop_loss = support * 0.98
                    
                    # Set take profit at resistance or 2:1 risk-reward
                    if resistance > entry_price:
                        take_profit = resistance
                    else:
                        # Calculate based on 2:1 risk-reward ratio
                        risk = entry_price - stop_loss
                        take_profit = entry_price + (risk * 2)
                        
                elif signal_direction == "sell" and resistance > 0:
                    # For sell signals, set stop loss above resistance
                    stop_loss = resistance * 1.02
                    
                    # Set take profit at support or 2:1 risk-reward
                    if support > 0 and support < entry_price:
                        take_profit = support
                    else:
                        # Calculate based on 2:1 risk-reward ratio
                        risk = stop_loss - entry_price
                        take_profit = entry_price - (risk * 2)
                
                # Create signal object
                signal = {
                    'asset': asset,
                    'direction': signal_direction,
                    'strength': trend_strength,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now().isoformat(),
                    'indicators': {
                        'rsi': indicators.get('rsi', 0),
                        'macd': indicators.get('macd', 0),
                        'trend': trend_direction
                    }
                }
                
                # Add to signals
                signals[asset] = signal
                
                # Track signal history
                if asset not in self.signal_history:
                    self.signal_history[asset] = []
                self.signal_history[asset].append({
                    'direction': signal_direction,
                    'strength': trend_strength,
                    'price': entry_price,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only recent signals in history (last 30)
                if len(self.signal_history[asset]) > 30:
                    self.signal_history[asset] = self.signal_history[asset][-30:]
            
            # Save signals to file
            filename = f"{self.signal_dir}/trading_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2)
            logger.info(f"Trading signals saved to {filename}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return {}
    
    def evaluate_signal_performance(self, signals, market_data_current, market_data_previous=None):
        """Evaluate the performance of previous trading signals"""
        if not market_data_previous:
            return {}
            
        evaluation = {}
        
        try:
            # Get previous signals
            prev_signal_files = sorted(os.listdir(self.signal_dir))
            if not prev_signal_files or len(prev_signal_files) < 2:
                return {}
            
            # Load previous signals (second to last file)
            prev_signal_file = prev_signal_files[-2]
            with open(f"{self.signal_dir}/{prev_signal_file}", 'r') as f:
                prev_signals = json.load(f)
            
            # For each previous signal
            for asset, signal in prev_signals.items():
                # Skip if we don't have current data for this asset
                if asset not in signals:
                    continue
                    
                # Get prices
                entry_price = signal.get('entry_price', 0)
                current_price = signals[asset].get('entry_price', 0)
                
                if entry_price == 0 or current_price == 0:
                    continue
                    
                # Calculate profit/loss
                direction = signal.get('direction', 'buy')
                
                if direction == 'buy':
                    pnl_percent = ((current_price / entry_price) - 1) * 100
                else:  # sell
                    pnl_percent = ((entry_price / current_price) - 1) * 100
                
                # Determine if signal was correct
                is_correct = (direction == 'buy' and pnl_percent > 0) or (direction == 'sell' and pnl_percent > 0)
                
                # Add to evaluation
                evaluation[asset] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'pnl_percent': pnl_percent,
                    'is_correct': is_correct
                }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating signal performance: {str(e)}")
            return {}