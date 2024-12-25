import logging
import sys
from datetime import datetime
from typing import Optional

# Import necessary modules
from signal_generator import AdvancedSignalGenerator
from data_source import DataSource
from trading_executor import TradingExecutor
from risk_manager import RiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def initialize_trading_bot():
    """
    Initialize trading bot components with error handling
    
    Returns:
        tuple: Initialized components or None
    """
    try:
        # Initialize data source
        data_source = DataSource()
        
        # Initialize risk manager
        risk_manager = RiskManager()
        
        # Initialize signal generator
        signal_generator = AdvancedSignalGenerator(data_source)
        
        # Initialize trading executor
        trading_executor = TradingExecutor()
        
        return data_source, signal_generator, risk_manager, trading_executor
    
    except Exception as init_error:
        logging.error(f"Bot initialization failed: {init_error}")
        return None

def run_trading_bot():
    """
    Main trading bot execution logic
    """
    logging.info("Trading Bot Starting...")
    
    # Initialize components
    components = initialize_trading_bot()
    
    if not components:
        logging.critical("Failed to initialize trading bot. Exiting.")
        sys.exit(1)
    
    data_source, signal_generator, risk_manager, trading_executor = components
    
    try:
        # Generate trading signal
        trading_signal = signal_generator.generate_signal()
        logging.info(f"Generated Trading Signal: {trading_signal}")
        
        # Assess risk
        risk_assessment = risk_manager.assess_risk(trading_signal)
        
        if risk_assessment['status'] == 'PROCEED':
            # Execute trade
            trade_result = trading_executor.execute_trade(
                signal=trading_signal, 
                risk_parameters=risk_assessment
            )
            
            logging.info(f"Trade Execution Result: {trade_result}")
        else:
            logging.warning(f"Trade Blocked: {risk_assessment['reason']}")
    
    except Exception as trading_error:
        logging.error(f"Trading bot encountered an error: {trading_error}")
    
    finally:
        logging.info("Trading Bot Execution Completed.")

def main():
    """
    Entry point for the trading bot
    """
    try:
        run_trading_bot()
    except KeyboardInterrupt:
        logging.info("Trading bot manually stopped.")
    except Exception as e:
        logging.critical(f"Unexpected error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()