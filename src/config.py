# List of KOL Twitter accounts to monitor
KOL_ACCOUNTS = [
    'VitalikButerin',
    'cz_binance',
    'aantonop',
    'CryptoHayes',
    'BarrySilbert',
    'APompliano',
    'elonmusk',
    'saylor',
    'CryptoYieldInfo',
    'DefiIgnas',
    # Add more accounts here
]

# Crypto terms and projects to track
CRYPTO_TERMS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "binance", "bnb",
    "cardano", "ada", "xrp", "dogecoin", "doge", "polkadot", "dot", "avalanche",
    "avax", "chainlink", "link", "litecoin", "ltc", "polygon", "matic", "shiba",
    "defi", "nft", "blockchain", "crypto", "token", "coin", "wallet", "exchange",
    "mining", "staking", "yield", "airdrop", "ico", "altcoin", "bull", "bear",
    "hodl", "fud", "fomo", "dex", "cex", "dao", "web3", "metaverse", "memecoin",
    # Add more terms here
]

# Posting schedule (in hours)
MARKET_ANALYSIS_INTERVAL = 4  # Post market analysis every 4 hours
DATA_COLLECTION_INTERVAL = 1   # Collect data every hour
MENTION_CHECK_INTERVAL = 10    # Check mentions every 10 minutes (in minutes)

# OpenAI model to use
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4" if you have access

# Market data settings
TOP_COINS_LIMIT = 100  # Number of top coins to track
WHALE_TX_MIN_VALUE = 1000000  # Minimum USD value for whale transactions