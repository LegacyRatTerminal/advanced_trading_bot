import numpy as np

class RiskManager:
    def __init__(self, config):
        # Trade Allocation
        self.MAX_ACCOUNT_ALLOCATION = 0.30  # 30% of account per trade
        
        # Leverage
        self.LEVERAGE = 20  # 20x leverage
        
        # Stop Loss and Profit Targets
        self.MAX_STOP_LOSS = 0.01  # 1% maximum loss
        self.MIN_PROFIT_TARGET = 0.01  # 1% minimum profit
        self.MAX_PROFIT_TARGET = 0.05  # 5% maximum profit

    def calculate_position_size(self, total_balance, current_price):
        """
        Calculate position size based on risk parameters
        """
        # Ensure total_balance is a float
        total_balance = float(total_balance)
        
        # Calculate maximum trade value
        max_trade_value = total_balance * self.MAX_ACCOUNT_ALLOCATION
        
        # Calculate position size with leverage
        # Ensure current_price is a float
        current_price = float(current_price)
        
        # Calculate position size
        position_size = (max_trade_value * self.LEVERAGE) / current_price
        
        return position_size
        
    def dynamic_stop_loss(self, trade_info, current_price):
        """
        Advanced dynamic stop-loss mechanism
        """
        entry_price = trade_info['entry_price']
        trade_type = trade_info['type']

        # Calculate current profit percentage
        if trade_type == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        # Dynamic stop-loss stages
        stages = [
            {'profit_threshold': 0.01, 'stop_loss_level': entry_price * 1.001},  # Break-even + 0.1%
            {'profit_threshold': 0.03, 'stop_loss_level': entry_price * 1.01},   # Lock in 1% profit
        ]

        # Find appropriate stop-loss level
        stop_loss_price = entry_price * (1 - self.MAX_STOP_LOSS)
        for stage in stages:
            if profit_pct >= stage['profit_threshold']:
                stop_loss_price = max(stop_loss_price, stage['stop_loss_level'])

        return stop_loss_price

    def intelligent_profit_taking(self, trade_info, current_price):
        """
        Advanced profit-taking strategy
        """
        entry_price = trade_info['entry_price']
        trade_type = trade_info['type']

        # Calculate current profit percentage
        if trade_type == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        # Profit-taking stages
        if profit_pct >= self.MAX_PROFIT_TARGET:
            return 'full_profit'
        elif profit_pct >= self.MIN_PROFIT_TARGET:
            return 'partial_profit'
        
        return 'hold'

    def should_exit_trade(self, trade_info, current_price):
        """
        Comprehensive trade exit decision
        """
        # Check stop-loss
        entry_price = trade_info['entry_price']
        trade_type = trade_info['type']

        # Calculate profit/loss percentage
        if trade_type == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        # Stop-loss condition
        if profit_pct <= -self.MAX_STOP_LOSS:
            return True

        # Profit-taking condition
        profit_action = self.intelligent_profit_taking(trade_info, current_price)
        return profit_action in ['full_profit', 'partial_profit']