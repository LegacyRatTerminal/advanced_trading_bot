# config/settings.py
class TradingConfig:
    # Exchange Configuration
    EXCHANGE = 'binance'
    API_KEY = '5UnHUp24PBWhfG43cJgMqkWUYVMUkJjJMBOWjpjCz3BLUEIzdQJnK6MaBb5X3anp'
    SECRET_KEY = 'kLIELdxpfyG33G50lEFo8XbwGy6gaixI0BOp2R3KWPEPHen2xKdF6iuHFFWIlmPQ'

    # Trading Parameters
    SYMBOLS = ['XRP/USDT', 'ADA/USDT', 'LINK/USDT', 'AAVE/USDT', 'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'EGLD/USDT', 'ENJ/USDT', 'FTM/USDT', 'INJ/USDT', 'MANA/USDT', 'UNI/USDT', 'AGLD/USDT']
    TIMEFRAMES = ['15m', '5m', '1h']

    # Risk Management
    MAX_LOSS_THRESHOLD = 0.01  # 1%
    MIN_PROFIT_TARGET = 0.01   # 1%
    MAX_PROFIT_TARGET = 0.05   # 5%
    LEVERAGE = 10

    # Machine Learning Configuration
    ML_MODELS = [
        'random_forest',
        'gradient_boosting'
    ]

    # Telegram Notification
    TELEGRAM_TOKEN = '7892769491:AAEVZuT2pQmL6034tfUt8EiV1DLDDX-XxCw'
    TELEGRAM_CHAT_ID = '7917402606'