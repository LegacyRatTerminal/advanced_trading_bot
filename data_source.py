import ccxt
import pandas as pd
import logging
from typing import List, Dict, Optional

class DataSource:
    def __init__(
        self, 
        exchange_name: str = 'binance', 
        config: Dict = None
    ):
        """
        Initialize cryptocurrency data source
        
        Args:
            exchange_name (str): Name of the exchange
            config (dict): Exchange configuration
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            # Dynamic exchange selection
            exchange_class = getattr(ccxt, exchange_name)
            
            # Initialize exchange
            self.exchange = exchange_class(config or {})
            
            # Load market data
            self.exchange.load_markets()
        
        except Exception as e:
            self.logger.error(f"Exchange initialization error: {e}")
            raise
    
    def get_historical_prices(
        self, 
        symbol: str = 'BTC/USDT', 
        timeframe: str = '1d', 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch historical price data with robust error handling
        
        Args:
            symbol (str): Trading pair
            timeframe (str): Candle timeframe
            limit (int): Number of candles
        
        Returns:
            DataFrame with price data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        
        except Exception as e:
            self.logger.error(f"Historical price fetch error for {symbol}: {e}")
            return pd.DataFrame()