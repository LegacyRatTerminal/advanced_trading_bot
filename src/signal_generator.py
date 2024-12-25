# src/signal_generator.py
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

class SignalGenerator:
    def __init__(self, config, exchange):
        self.config = config
        self.exchange = exchange
        self.models = {}
        self.scalers = {}
        self._initialize_and_train_models()

    def _create_advanced_features(self, df):
        """
        Create more sophisticated features
        """
        # Create a copy to avoid SettingWithCopyWarning
        features_df = df.copy()
        
        # Trend indicators
        features_df['sma_50'] = features_df['close'].rolling(window=50).mean()
        features_df['sma_200'] = features_df['close'].rolling(window=200).mean()
        
        # Momentum indicators
        features_df['rsi'] = ta.momentum.RSIIndicator(close=features_df['close']).rsi()
        features_df['macd'] = ta.trend.MACD(close=features_df['close']).macd()
        features_df['stoch_rsi'] = ta.momentum.StochRSIIndicator(close=features_df['close']).stochrsi()
        
        # Volatility indicators
        features_df['bbands_high'] = ta.volatility.BollingerBands(close=features_df['close']).bollinger_hband()
        features_df['bbands_low'] = ta.volatility.BollingerBands(close=features_df['close']).bollinger_lband()
        
        # Price change features
        features_df['price_change_1d'] = features_df['close'].pct_change(periods=1)
        features_df['price_change_7d'] = features_df['close'].pct_change(periods=7)
        
        return features_df.dropna()

    def _initialize_and_train_models(self):
        """
        Advanced model training with feature selection and cross-validation
        """
        for symbol in self.config.SYMBOLS:
            try:
                # Fetch historical data
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.config.TIMEFRAMES[0], limit=2000)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Create advanced features
                features_df = self._create_advanced_features(df)
                
                # Create target variable (binary classification)
                # Use .loc to avoid SettingWithCopyWarning
                features_df.loc[:, 'target'] = np.where(
                    features_df['close'].pct_change(periods=1).shift(-1) > 0, 
                    1, 0
                )
                
                # Prepare features and target
                feature_columns = [
                    'sma_50', 'sma_200', 'rsi', 'macd', 'stoch_rsi', 
                    'bbands_high', 'bbands_low', 'price_change_1d', 'price_change_7d'
                ]
                X = features_df[feature_columns]
                y = features_df['target']
                
                # Check target distribution
                if len(np.unique(y)) < 2:
                    print(f"Skipping {symbol}: Insufficient target variation")
                    continue
                
                # Create model pipelines with feature selection
                models_to_train = {
                    'random_forest': Pipeline([
                        ('scaler', StandardScaler()),
                        ('feature_selection', SelectKBest(f_classif, k=5)),
                        ('classifier', RandomForestClassifier(
                            n_estimators=200, 
                            random_state=42, 
                            class_weight='balanced'
                        ))
                    ]),
                    'gradient_boosting': Pipeline([
                        ('scaler', StandardScaler()),
                        ('feature_selection', SelectKBest(f_classif, k=5)),
                        ('classifier', GradientBoostingClassifier(
                            n_estimators=200, 
                            random_state=42, 
                            learning_rate=0.1
                        ))
                    ])
                }
                
                # Train and evaluate models
                symbol_models = {}
                for name, model in models_to_train.items():
                    # Perform cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=5)
                    
                    # Fit on full dataset
                    model.fit(X, y)
                    
                    # Store model
                    symbol_models[name] = {
                        'model': model,
                        'cv_score': cv_scores.mean()
                    }
                    
                    # Print detailed performance
                    print(f"Model performance for {symbol} - {name}:")
                    print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Store models for this symbol
                self.models[symbol] = symbol_models
                
            except Exception as e:
                print(f"Error training models for {symbol}: {e}")

    def generate_signals(self, symbol, timeframe):
        """
        Generate trading signals using machine learning
        """
        # Check if models exist for the symbol
        if symbol not in self.models:
            print(f"No models trained for {symbol}")
            return {}

        # Fetch recent data
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Create features
        features_df = self._create_advanced_features(df)
        
        # Get most recent features
        feature_columns = [
            'sma_50', 'sma_200', 'rsi', 'macd', 'stoch_rsi', 
            'bbands_high', 'bbands_low', 'price_change_1d', 'price_change_7d'
        ]
        recent_features = features_df[feature_columns].iloc[-1:]
        
        # Predict using multiple models
        signals = {}
        for name, model_info in self.models[symbol].items():
            model = model_info['model']
            prediction = model.predict(recent_features)
            signals[name] = prediction[0]
        
        return signals