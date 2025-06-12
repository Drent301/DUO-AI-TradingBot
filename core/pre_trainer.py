# core/pre_trainer.py
import logging
import os
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import talib # Direct import of talib
import dotenv
import sqlite3 # For test block
from sklearn.preprocessing import MinMaxScaler # For normalization

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import ParamsManager for use in PreTrainer class
try:
    from core.params_manager import ParamsManager
except ImportError:
    ParamsManager = None # Allow testing even if ParamsManager is not fully available initially

logger = logging.getLogger(__name__)
# Ensure logger has a handler, e.g., if running this file directly for testing
# Moved sys import to the top of the file for standard practice,
# but it's only used in __main__. If this file is imported, sys might not be needed by the logger immediately.
import sys
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
logger.setLevel(logging.INFO)


MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
PRETRAIN_LOG_FILE = os.path.join(MEMORY_DIR, 'pre_train_log.json')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'models')
# CNN_MODEL_PATH is now dynamic per pattern, e.g., cnn_model_bullFlag.pth
# CNN_SCALER_PATH is also dynamic per pattern, e.g., scaler_params_bullFlag.json

os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = None
        self.flatten_size = None
        self._num_classes_for_fc_init = num_classes # Store for dynamic init

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        if self.fc1 is None:
            self.flatten_size = x.shape[1] * x.shape[2]
            self.fc1 = nn.Linear(self.flatten_size, self._num_classes_for_fc_init)
            self.fc1.to(x.device)
        x = x.view(-1, self.flatten_size)
        x = self.fc1(x)
        return x

class PreTrainer:
    def __init__(self, params_manager_instance: Optional[ParamsManager] = None):
        if params_manager_instance:
            self.params_manager = params_manager_instance
        elif ParamsManager: # Check if ParamsManager was imported successfully
            self.params_manager = ParamsManager()
        else:
            logger.error("ParamsManager is not available. PreTrainer cannot function without it.")
            raise ImportError("ParamsManager could not be imported or provided.")
        logger.info("PreTrainer geïnitialiseerd.")
        # Initialize CNNPatterns here if it's frequently used or for direct pattern generation
        # For now, it's instantiated in prepare_training_data if needed for labels.

    async def fetch_historical_data_from_db(self, symbol: str, timeframe: str, limit: int = 2000) -> pd.DataFrame:
        logger.info(f"Fetching historical data for {symbol} ({timeframe}) from DB, limit {limit}...")
        # This path needs to be correctly configured for your Freqtrade setup
        # FREQTRADE_DB_PATH defined globally for the module or passed appropriately
        conn = None
        try:
            # Correctly use await for the async thread operation
            conn = await asyncio.to_thread(sqlite3.connect, FREQTRADE_DB_PATH)
            # Parameterized query for safety and correctness
            query = f"SELECT * FROM candles_{symbol.replace('/', '_')}_{timeframe} ORDER BY date DESC LIMIT ?;"
            # Use pandas to execute the query with parameters
            df = pd.read_sql_query(query, conn, params=(limit,))
            if df.empty:
                logger.warning(f"No data found for {symbol} ({timeframe}) in database {FREQTRADE_DB_PATH}.")
                return pd.DataFrame()

            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'date': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], unit='s') # Assuming date is stored as Unix timestamp in seconds
            df.set_index('Date', inplace=True)
            df.sort_index(ascending=True, inplace=True) # Ensure data is chronological
            # Store metadata as attributes
            df.attrs['timeframe'] = timeframe
            df.attrs['pair'] = symbol
            logger.info(f"Historical data fetched for {symbol} ({timeframe}): {len(df)} candles.")
            return df
        except sqlite3.Error as e:
            logger.error(f"Database error while fetching data for {symbol} ({timeframe}): {e}")
            return pd.DataFrame() # Return empty DataFrame on error
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol} ({timeframe}): {e}")
            return pd.DataFrame()
        finally:
            if conn:
                await asyncio.to_thread(conn.close)


    async def prepare_training_data(self, dataframe: pd.DataFrame, pattern_name: str = 'bullFlag') -> pd.DataFrame:
        logger.info(f"Preparing training data for pattern '{pattern_name}'...")
        if dataframe.empty:
            logger.warning("Empty dataframe received in prepare_training_data.")
            return dataframe

        df_copy = dataframe.copy()

        # Add common indicators
        df_copy['rsi'] = talib.RSI(df_copy['Close'].values, timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(df_copy['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df_copy['macd'] = macd
        df_copy['macdsignal'] = macdsignal
        df_copy['macdhist'] = macdhist

        # Labeling logic (example for 'bullFlag', can be expanded)
        # This section needs to be highly adaptable based on `pattern_name`
        if pattern_name == 'bullFlag':
            # Example: A simple bull flag might be a short consolidation after a sharp rise.
            # For ML, we need a more robust definition or use a pre-labeled dataset / different approach.
            # Placeholder: label based on future profit as in the original example
            future_N_candles = self.params_manager.get_param('future_N_candles_for_label', default=20)
            profit_threshold_pct = self.params_manager.get_param('profit_threshold_for_label', default=0.02)
            loss_threshold_pct = self.params_manager.get_param('loss_threshold_for_label', default=-0.01)

            df_copy['future_high'] = df_copy['High'].shift(-future_N_candles)
            df_copy['future_low'] = df_copy['Low'].shift(-future_N_candles)

            profit_target_met = (df_copy['future_high'] - df_copy['Close']) / df_copy['Close'] >= profit_threshold_pct
            loss_target_met = (df_copy['future_low'] - df_copy['Close']) / df_copy['Close'] <= loss_threshold_pct

            df_copy[f'{pattern_name}_label'] = np.where(profit_target_met & ~loss_target_met, 1, 0)
            df_copy.drop(columns=['future_high', 'future_low'], inplace=True, errors='ignore')

        elif pattern_name == 'bearTrap': # Example for a different pattern
            # A bear trap: price breaks below support, then sharply reverses upwards.
            # Labeling would be complex: identify support, break, then reversal.
            # This is where a dedicated labeling function per pattern would be essential.
            logger.warning(f"Labeling for pattern '{pattern_name}' is not fully implemented. Using placeholder zeros.")
            df_copy[f'{pattern_name}_label'] = 0 # Placeholder
        else:
            logger.warning(f"Unknown pattern_name '{pattern_name}' for labeling. No labels generated.")
            return pd.DataFrame() # Or handle as appropriate

        # Define feature columns (ensure these are common across patterns or handled dynamically)
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'macd', 'macdsignal', 'macdhist']

        cols_to_drop_for_na = [f'{pattern_name}_label'] + self.feature_columns
        df_copy.dropna(subset=cols_to_drop_for_na, inplace=True)

        logger.info(f"Training data prepared for '{pattern_name}' with {len(df_copy)} samples. Features: {self.feature_columns}")
        return df_copy

    async def train_ai_models(self, training_data: pd.DataFrame, pattern_name: str = 'bullFlag'):
        logger.info(f"Starting training of PyTorch AI model for pattern '{pattern_name}' with {len(training_data)} samples...")

        if training_data.empty or f'{pattern_name}_label' not in training_data.columns:
            logger.warning(f"No training data or labels for pattern '{pattern_name}'. Training skipped.")
            return

        if not hasattr(self, 'feature_columns') or not self.feature_columns:
            logger.error("Feature columns not set (e.g., via prepare_training_data). Cannot train model.")
            return

        sequence_length = self.params_manager.get_param('sequence_length_cnn', default=30)
        num_epochs = self.params_manager.get_param('num_epochs_cnn', default=10)
        batch_size = self.params_manager.get_param('batch_size_cnn', default=32)
        learning_rate = self.params_manager.get_param('learning_rate_cnn', default=0.001)

        X_list, y_list = [], []
        for i in range(len(training_data) - sequence_length):
            X_list.append(training_data[self.feature_columns].iloc[i:i + sequence_length].values)
            y_list.append(training_data[f'{pattern_name}_label'].iloc[i + sequence_length - 1])

        if not X_list:
            logger.warning(f"Not enough data to create sequences for '{pattern_name}'. Training skipped.")
            return

        X_np = np.array(X_list)
        y_np = np.array(y_list)

        # Normalization using MinMaxScaler (applied per feature across all sequences)
        # Reshape X to 2D for scaling: (num_samples * sequence_length, num_features)
        X_reshaped = X_np.reshape(-1, X_np.shape[-1])
        scaler = MinMaxScaler()
        X_normalized_reshaped = scaler.fit_transform(X_reshaped)
        # Reshape back to 3D: (num_samples, sequence_length, num_features)
        X_normalized = X_normalized_reshaped.reshape(X_np.shape)

        # Save scaler parameters
        scaler_params = {
            'feature_columns': self.feature_columns,
            'min_': scaler.min_.tolist(),
            'scale_': scaler.scale_.tolist()
        }
        scaler_path = os.path.join(MODELS_DIR, f'scaler_params_{pattern_name}.json')
        with open(scaler_path, 'w') as f:
            json.dump(scaler_params, f, indent=4)
        logger.info(f"Scaler parameters for '{pattern_name}' saved to {scaler_path}")

        # Tensor Conversion: PyTorch Conv1d expects (batch, channels, seq_len)
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32).permute(0, 2, 1)
        y_tensor = torch.tensor(y_np, dtype=torch.long) # CrossEntropyLoss expects Long type

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_channels = X_tensor.shape[1] # Number of features
        model = SimpleCNN(input_channels=input_channels, num_classes=2) # Assuming binary classification (pattern vs no pattern)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        logger.info(f"Starting training loop for '{pattern_name}' for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for data, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] for '{pattern_name}', Avg Loss: {avg_epoch_loss:.4f}")

        model_save_path = os.path.join(MODELS_DIR, f'cnn_model_{pattern_name}.pth')
        try:
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"PyTorch model for '{pattern_name}' trained and saved to {model_save_path}")
        except Exception as e:
            logger.error(f"Error saving PyTorch model for '{pattern_name}': {e}")
            return

        await self._log_pretrain_activity(pattern_name, len(X_np), model_save_path, scaler_path)

    async def _log_pretrain_activity(self, pattern_name: str, data_size: int, model_path: str, scaler_path: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "pattern_name": pattern_name,
            "data_size": data_size,
            "status": "completed_pytorch_training",
            "model_path_saved": model_path,
            "scaler_params_path_saved": scaler_path
        }
        logs = []
        # Similar loading and appending logic as before, ensuring atomicity if possible or handling races
        # For simplicity, using basic load/append/write here
        try:
            if os.path.exists(PRETRAIN_LOG_FILE) and os.path.getsize(PRETRAIN_LOG_FILE) > 0:
                with open(PRETRAIN_LOG_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content: logs = json.loads(content)
                    if not isinstance(logs, list): logs = [logs]
            logs.append(entry)
            with open(PRETRAIN_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging pre-train activity: {e}")

    async def run_pretraining_pipeline(self, symbol: str, timeframe: str, strategy_id: str, params_manager_instance: Optional[ParamsManager] = None):
        # Ensure self.params_manager is correctly set
        if params_manager_instance:
            self.params_manager = params_manager_instance
        elif not hasattr(self, 'params_manager') and ParamsManager: # If not set in __init__ (e.g. direct call)
             self.params_manager = ParamsManager()
        elif not hasattr(self, 'params_manager'):
            logger.error("ParamsManager not available in run_pretraining_pipeline.")
            return

        logger.info(f"Starting pre-training pipeline for {symbol} ({timeframe}, strategy: {strategy_id})...")

        # 1. Fetch Data
        historical_df = await self.fetch_historical_data_from_db(symbol, timeframe, limit=2000) # Example limit
        if historical_df.empty:
            logger.error(f"Cannot run pre-training: no historical data for {symbol} ({timeframe}).")
            return

        # 2. Define patterns to train models for
        # This could come from config or be dynamic
        patterns_to_train = self.params_manager.get_param("trainablePatterns", strategy_id=strategy_id, default=['bullFlag'])
        if not patterns_to_train:
            logger.info("No patterns specified for training in params.json (trainablePatterns). Skipping model training.")
            return

        logger.info(f"Found patterns to train: {patterns_to_train}")

        for pattern_name in patterns_to_train:
            logger.info(f"Processing pattern: {pattern_name} for {symbol} ({timeframe})")
            # 3. Prepare Data for the specific pattern
            # This will add a '{pattern_name}_label' column
            processed_df = await self.prepare_training_data(historical_df, pattern_name=pattern_name)
            if processed_df.empty or f'{pattern_name}_label' not in processed_df.columns:
                logger.error(f"Cannot train for pattern '{pattern_name}': data preparation failed or no labels generated.")
                continue # Skip to next pattern

            # 4. Train Model for the specific pattern
            await self.train_ai_models(processed_df, pattern_name=pattern_name)

        logger.info(f"Pre-training pipeline completed for {symbol} ({timeframe}).")


# Global FREQTRADE_DB_PATH for the test block
FREQTRADE_DB_PATH = "test_freqtrade_pretrainer.sqlite" # In-memory or test file

# Test block
if __name__ == "__main__":
    # import sys # Already imported at the top for logger config
    # Configure logger for test output
    # BasicConfig should ideally be called only once.
    # If it was called when the logger was first defined, this might be redundant or cause issues.
    # However, for a direct script run, it's often placed here.
    # To be safe, check if handlers are already present.
    if not logging.getLogger().handlers: # Check root logger
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])

    dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

    # Test-specific FREQTRADE_DB_PATH
    # This will be created in the same directory as the script when run directly
    # FREQTRADE_DB_PATH = "test_freqtrade_pretrainer.sqlite" # Defined globally for the module now

    def setup_dummy_db_for_pretrainer():
        if os.path.exists(FREQTRADE_DB_PATH):
            os.remove(FREQTRADE_DB_PATH)
        conn = sqlite3.connect(FREQTRADE_DB_PATH)
        cursor = conn.cursor()
        # Create a table for ETH/USDT 5m - names must match Freqtrade's convention
        # Store date as Unix timestamp (seconds) as Freqtrade does
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles_ETH_USDT_5m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date INTEGER UNIQUE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            );
        """)
        # Populate with some dummy data
        now_ts = int(datetime.now().timestamp())
        for i in range(1000): # More data for sequence generation
            # Timestamps should be in seconds and unique
            date_ts = now_ts - (1000 - i) * 300 # 300 seconds = 5 minutes
            open_ = 100 + i * 0.01 + np.random.rand() * 0.1
            close_ = open_ + (np.random.rand() - 0.5) * 0.2
            high_ = max(open_, close_) + np.random.rand() * 0.1
            low_ = min(open_, close_) - np.random.rand() * 0.1
            volume = 1000 + np.random.rand() * 100
            try:
                cursor.execute("INSERT INTO candles_ETH_USDT_5m (date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)",
                               (date_ts, open_, high_, low_, close_, volume))
            except sqlite3.IntegrityError: # Skip if date already exists (should not happen with this logic)
                pass
        conn.commit()
        conn.close()
        logger.info(f"Dummy database for pretrainer test created at {FREQTRADE_DB_PATH} with ETH/USDT 5m data.")

    async def run_test_pre_trainer():
        setup_dummy_db_for_pretrainer()

        # Clean up old model/scaler files for a clean test run
        test_pattern_name = 'bullFlag' # Default pattern used in pipeline test
        model_file_to_check = os.path.join(MODELS_DIR, f'cnn_model_{test_pattern_name}.pth')
        scaler_file_to_check = os.path.join(MODELS_DIR, f'scaler_params_{test_pattern_name}.json')

        if os.path.exists(model_file_to_check):
            os.remove(model_file_to_check)
            logger.info(f"Removed old test model: {model_file_to_check}")
        if os.path.exists(scaler_file_to_check):
            os.remove(scaler_file_to_check)
            logger.info(f"Removed old test scaler: {scaler_file_to_check}")

        # Initialize ParamsManager for the test if needed by PreTrainer's __init__
        # or if run_pretraining_pipeline needs it explicitly.
        # The PreTrainer now has a fallback if no PM is passed to __init__.
        # However, run_pretraining_pipeline also has a fallback.
        # For clarity, let's pass one.
        pm_instance = None
        if ParamsManager:
            pm_instance = ParamsManager()
            # Set trainablePatterns for the test if it's empty or not what we want
            current_trainable = pm_instance.get_param("trainablePatterns", strategy_id="DUOAI_Strategy_PretrainTest")
            if not current_trainable or test_pattern_name not in current_trainable:
                 await pm_instance.set_param("trainablePatterns", [test_pattern_name], strategy_id="DUOAI_Strategy_PretrainTest")
                 logger.info(f"Set 'trainablePatterns' to {[test_pattern_name]} for test strategy.")
        else:
            logger.warning("ParamsManager not imported, test might be limited.")


        pre_trainer = PreTrainer(params_manager_instance=pm_instance) # Pass instance
        test_symbol = "ETH/USDT"
        test_timeframe = "5m"
        # Use a distinct strategy_id for testing to avoid conflicts with live params.json
        test_strategy_id = "DUOAI_Strategy_PretrainTest"

        print("\n--- Test PreTrainer Pipeline (PyTorch CNN) ---")
        # Pass the params_manager_instance to the pipeline as well
        await pre_trainer.run_pretraining_pipeline(test_symbol, test_timeframe, test_strategy_id, params_manager_instance=pm_instance)

        print(f"\nCheck logs in {PRETRAIN_LOG_FILE}")
        if os.path.exists(model_file_to_check):
            print(f"SUCCES: PyTorch CNN model opgeslagen op: {model_file_to_check}")
            print(f"Modelgrootte: {os.path.getsize(model_file_to_check)} bytes")
        else:
            print(f"FOUT: PyTorch CNN model NIET opgeslagen op: {model_file_to_check}")

        if os.path.exists(scaler_file_to_check):
            print(f"SUCCES: CNN scaler parameters opgeslagen op: {scaler_file_to_check}")
        else:
            print(f"FOUT: CNN scaler parameters NIET opgeslagen op: {scaler_file_to_check}")

        # Cleanup dummy DB
        if os.path.exists(FREQTRADE_DB_PATH):
            os.remove(FREQTRADE_DB_PATH)
            logger.info(f"Dummy database {FREQTRADE_DB_PATH} removed.")

    try:
        asyncio.run(run_test_pre_trainer())
    except ImportError as e:
        logger.error(f"Test run failed due to import error: {e}. Ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the test run: {e}", exc_info=True)
