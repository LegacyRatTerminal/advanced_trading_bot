import asyncio
import ccxt
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .risk_management import RiskManager
from .signal_generator import SignalGenerator
from .utils import Logger, TelegramNotifier
from .advanced_signal_generator import AdvancedSignalGenerator

class AdvancedTradingBot:
    def __init__(self, config):
        # Configuration
        self.config = config
        
        # Logging
        self.logger = Logger.setup_logging()
        self.notifier = TelegramNotifier(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
        
        # Exchange Setup
        self.exchange = self._setup_exchange()
        
        # Core Components
        self.risk_manager = RiskManager(config)
        self.signal_generator = AdvancedSignalGenerator(config, self.exchange)
        
        # Trade Management
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance Tracking
        self.total_profit = 0
        self.total_trades = 0
        self.winning_trades = 0

    def _setup_exchange(self):
        """
        Set up the cryptocurrency exchange with advanced configurations
        """
        try:
            exchange_class = getattr(ccxt, self.config.EXCHANGE)
            exchange = exchange_class({
                'apiKey': self.config.API_KEY,
                'secret': self.config.SECRET_KEY,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Use futures trading
                }
            })
            return exchange
        except Exception as e:
            self.logger.error(f"Exchange setup failed: {e}")
            raise

    async def run(self):
        """
        Advanced trading bot main loop with comprehensive error handling
        """
        await self.notifier.send_message("🚀 Advanced Trading Bot Initialized")
        self.logger.info("Trading bot started")

        try:
            while True:
                # Comprehensive trading cycle
                await self._trading_cycle()
                
                # Performance monitoring and logging
                await self._log_performance()
                
                # Clean up stale trades
                self._cleanup_stale_trades()
                
                # Wait before next cycle
                await asyncio.sleep(60)  # Adjust cycle time as needed

        except Exception as critical_error:
            error_message = f"🚨 Critical Bot Error: {critical_error}"
            self.logger.critical(error_message)
            await self.notifier.send_message(error_message)
            raise

    async def _trading_cycle(self):
        """
        Comprehensive trading cycle for multiple symbols
        """
        for symbol in self.config.SYMBOLS:
            try:
                # Generate trading signals
                signals = self.signal_generator.generate_signals(symbol)
                
                # Process signals and manage trades
                await self._process_signals(symbol, signals)
                
                # Monitor and manage existing trades
                await self._manage_active_trades(symbol)
                
            except Exception as symbol_error:
                error_msg = f"Error processing {symbol}: {symbol_error}"
                self.logger.error(error_msg)
                await self.notifier.send_message(error_msg)

    async def _process_signals(self, symbol: str, signals: Dict[str, int]):
        """
        Advanced signal processing with multi-model consensus
        """
        # Check if we already have 2 active trades
        if len(self.active_trades) >= 2:
            return

        if not signals:
            return

        # Consensus mechanism
        consensus = signals.get('consensus', 0)

        # Decision making
        if symbol not in self.active_trades:
            if consensus == 1:
                await self._execute_trade(symbol, 'buy')
            elif consensus == -1:
                await self._execute_trade(symbol, 'sell')

    async def _execute_trade(self, symbol: str, trade_type: str):
        """
        Advanced trade execution with risk management
        """
        try:
            # Fetch current market price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']

            # Calculate position size
            balance = self.exchange.fetch_balance()
            position_size = self.risk_manager.calculate_position_size(
                balance['total'], current_price
            )

            # Execute trade
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=trade_type,
                amount=position_size,
                params={'leverage': self.config.LEVERAGE}
            )

            # Track trade
            trade_info = {
                'symbol': symbol,
                'type': trade_type,
                'entry_price': current_price,
                'amount': position_size,
                'timestamp': self.exchange.milliseconds()
            }
            self.active_trades[symbol] = trade_info

            # Notification
            await self.notifier.send_message(
                f"📈 {trade_type.upper()} Trade Executed: {symbol} @ {current_price}"
            )

        except Exception as trade_error:
            self.logger.error(f"Trade execution error: {trade_error}")
            await self.notifier.send_message(f"🚨 Trade Execution Failed: {trade_error}")

    async def _manage_active_trades(self, symbol: str):
        """
        Advanced trade management with dynamic stop loss and take profit
        """
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        current_ticker = self.exchange.fetch_ticker(symbol)
        current_price = current_ticker['last']
    
        # Dynamic stop-loss calculation
        dynamic_stop_loss_price = self.risk_manager.dynamic_stop_loss(trade, current_price)

        # Check exit conditions
        if self.risk_manager.should_exit_trade(trade, current_price):
            await self._close_trade(symbol, trade)
        else:
            # Optional: Update trade with dynamic stop-loss
            trade['dynamic_stop_loss'] = dynamic_stop_loss_price

    async def _close_trade(self, symbol: str, trade: Dict[str, Any]):
        """
        Advanced trade closure with performance tracking
        """
        try:
            # Close position
            self.exchange.create_market_order(
                symbol=symbol,
                side='sell' if trade['type'] == 'buy' else 'buy',
                amount=trade['amount']
            )

            # Calculate profit
            current_ticker = self.exchange.fetch_ticker(symbol)
            profit = self._calculate_trade_profit(trade, current_ticker['last'])

            # Update performance metrics
            self.total_trades += 1
            self.total_profit += profit
            
            if profit > 0:
                self.winning_trades += 1

            # Remove from active trades
            del self.active_trades[symbol]

            # Logging and notification
            await self.notifier.send_message(
                f"🏁 Trade Closed: {symbol}\n"
                f"Entry: {trade['entry_price']}\n"
                f"Exit: {current_ticker['last']}\n"
                f"Profit: {profit:.2f}%"
            )

        except Exception as close_error:
            self.logger.error(f"Trade closure error: {close_error}")

    def _calculate_trade_profit(self, trade: Dict[str, Any], exit_price: float) -> float:
        """
        Calculate trade profit with precision
        """
        if trade['type'] == 'buy':
            return (exit_price - trade['entry_price']) / trade['entry_price'] * 100
        else:
            return (trade['entry_price'] - exit_price) / trade['entry_price'] * 100

    async def _log_performance(self):
        """
        Comprehensive performance logging
        """
        # Prevent division by zero
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
    
        performance_msg = (
            f"📊 Bot Performance:\n"
            f"Total Trades: {self.total_trades}\n"
            f"Winning Trades: {self.winning_trades}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Total Profit: {self.total_profit:.2f}%"
        )
    
        self.logger.info(performance_msg)
        await self.notifier.send_message(performance_msg)

    def _cleanup_stale_trades(self):
        """
        Remove trades that have been open for too long
        """
        current_time = self.exchange.milliseconds()
        stale_trades = [
            symbol for symbol, trade in self.active_trades.items()
            if current_time - trade['timestamp'] > 24 * 60 * 60 * 1000  # 24 hours
        ]

        for symbol in stale_trades:
            del self.active_trades[symbol]
            self.logger.warning(f"Removed stale trade for {symbol}")