#!/bin/bash

# 自动修复加密分析机器人常见问题的脚本
echo "开始自动修复加密分析机器人..."

# 切换到项目目录
cd ~/crypto_analysis_bot

# 激活虚拟环境
source venv/bin/activate

# 确保所有必要的目录存在
mkdir -p data logs

# 安装/更新依赖项
echo "更新依赖项..."
pip install --upgrade pip
pip install --upgrade openai tweepy pandas numpy matplotlib scikit-learn transformers nltk requests python-dotenv schedule

# 下载NLTK数据
echo "下载NLTK数据..."
python setup_nltk.py

# 确保NLTK_DATA环境变量在服务文件中设置
sudo grep -q "NLTK_DATA" /etc/systemd/system/crypto-bot.service
if [ $? -ne 0 ]; then
    echo "在服务文件中添加NLTK_DATA环境变量..."
    sudo sed -i '/WorkingDirectory=/a Environment="NLTK_DATA=/home/ubuntu/nltk_data"' /etc/systemd/system/crypto-bot.service
    sudo systemctl daemon-reload
fi

# 为服务用户确保NLTK数据路径可访问
chmod -R 755 ~/nltk_data

# 测试机器人的各个组件
echo "测试机器人组件..."
python -c "from src.modules.nlp_analyzer import NLPAnalyzer; analyzer = NLPAnalyzer(); print('NLP分析器初始化成功')"
python -c "from src.modules.market_data import MarketData; data = MarketData(); print('市场数据组件初始化成功')"
python -c "from src.modules.blockchain_data import BlockchainData; blockchain = BlockchainData(); print('区块链数据组件初始化成功')"

echo "自动修复完成，重启机器人服务..."
sudo systemctl restart crypto-bot.service
sudo systemctl status crypto-bot.service --no-pager

echo "查看日志以确认机器人运行状态:"
echo "sudo journalctl -u crypto-bot.service -f"