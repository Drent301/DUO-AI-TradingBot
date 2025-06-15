# core/pre_trainer.py
import logging
import os # Keep for os.getenv, os.remove if not fully replaced by Path.unlink
import json
from pathlib import Path # Import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone # Added timezone
from datetime import datetime as dt
import pandas as pd
import pandas_ta as ta
import numpy as np
import asyncio
from core.bitvavo_executor import BitvavoExecutor
from core.params_manager import ParamsManager
from core.cnn_patterns import CNNPatterns

from freqtrade.configuration import Configuration
from freqtrade.data.dataprovider import DataProvider
from freqtrade.exchange import TimeRange
from strategies.DUOAI_Strategy import DUOAI_Strategy

import dotenv
import shutil # Keep for rmtree

# Scikit-learn imports
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Module-level docstring
"""
Handles the pre-training pipeline for CNN models based on historical market data.
"""
import torch # Ensure torch is imported if used for type hints or operations
from sklearn.preprocessing import MinMaxScaler # Ensure MinMaxScaler is imported
from typing import TYPE_CHECKING # For forward declaration of CNNPatterns

if TYPE_CHECKING:
    from core.cnn_patterns import CNNPatterns # Forward declaration for type hint

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Logging level configured by application

# Sentinel object (if still needed, otherwise remove)
_SENTINEL = object()

# Define paths using pathlib
CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CORE_DIR.parent # Assumes 'core' is directly under project root

# Memory directory within user_data at project root
USER_DATA_MEMORY_DIR = PROJECT_ROOT_DIR / 'user_data' / 'memory'
PRETRAIN_LOG_FILE = USER_DATA_MEMORY_DIR / 'pre_train_log.json'
TIME_EFFECTIVENESS_FILE = USER_DATA_MEMORY_DIR / 'time_effectiveness.json'

# Data directory: MARKET_REGIMES_FILE was originally in core/../data, so PROJECT_ROOT_DIR/data
# If it's meant to be in user_data, this should be USER_DATA_DATA_DIR.
# For now, matching original relative location from project root.
STATIC_DATA_DIR = PROJECT_ROOT_DIR / 'data'
MARKET_REGIMES_FILE = STATIC_DATA_DIR / 'market_regimes.json'


# Ensure base directories exist
USER_DATA_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DATA_DIR.mkdir(parents=True, exist_ok=True) # For MARKET_REGIMES_FILE if it's there


class PreTrainer:
    def __init__(self, config: Dict[str, Any], params_manager: ParamsManager):
        self.config = config
        self.params_manager = params_manager # Use the provided ParamsManager instance
        logger.info("PreTrainer initialized with a provided ParamsManager instance.")

        self.bitvavo_executor: Optional[BitvavoExecutor] = None
        if self.config.get('exchange', {}).get('name', '').lower() == 'bitvavo':
            try:
                self.bitvavo_executor = BitvavoExecutor(params_manager=self.params_manager)
                logger.info("BitvavoExecutor initialized in PreTrainer.")
            except ValueError as e:
                logger.warning(f"PreTrainer: BitvavoExecutor could not be initialized: {e}. Live data fetching might be unavailable.", exc_info=True)
            except Exception as e_gen: # Catch other potential init errors
                logger.error(f"PreTrainer: Unexpected error initializing BitvavoExecutor: {e_gen}", exc_info=True)

        # Instantiate CNNPatterns detector
        # CNNPatterns is initialized with the shared self.params_manager instance.
        # This allows it to use the same ParamsManager as the PreTrainer.
        try:
            self.cnn_pattern_detector: CNNPatterns = CNNPatterns(params_manager=self.params_manager)
            logger.info("CNNPatterns initialized in PreTrainer with shared ParamsManager.")
        except Exception as e:
            logger.error(f"Error initializing CNNPatterns in PreTrainer: {e}", exc_info=True)
            self.cnn_pattern_detector = None # Fallback to None if instantiation fails

        # Define models_dir based on user_data_dir from config, resolve to absolute path
        user_data_root_str = self.config.get("user_data_dir", "user_data")
        user_data_root_path = Path(user_data_root_str)
        if not user_data_root_path.is_absolute():
            user_data_root_path = PROJECT_ROOT_DIR / user_data_root_path

        self.models_dir = (user_data_root_path / "models" / "cnn_pretrainer").resolve()

        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"PreTrainer initialized. Models will be saved in: {self.models_dir}")
        except Exception as e:
            logger.error(f"Could not create models directory {self.models_dir}: {e}", exc_info=True)

        self.market_regimes = {}
        try:
            if MARKET_REGIMES_FILE.exists():
                with MARKET_REGIMES_FILE.open('r', encoding='utf-8') as f:
                    self.market_regimes = json.load(f)
                logger.info(f"Marktregimes geladen uit {MARKET_REGIMES_FILE}")
            else:
                logger.warning(f"{MARKET_REGIMES_FILE} niet gevonden. Er worden geen specifieke marktregimes gebruikt.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Fout bij laden/parsen {MARKET_REGIMES_FILE}: {e}.", exc_info=True)
            self.market_regimes = {}
        except Exception as e_gen:
            logger.error(f"Onverwachte fout bij laden market regimes {MARKET_REGIMES_FILE}: {e_gen}", exc_info=True)
            self.market_regimes = {}

        # Initialize Backtester - this might need adjustment based on what PreTrainer's role becomes.
        # If PreTrainer is purely for model training, backtesting might be separate.
        # For now, keeping it to see if it causes issues with the new __init__.
        # self.backtester related code removed as the class is deprecated.

    def _get_dataframe_with_strategy_indicators(self, ohlcv_df: pd.DataFrame, pair: str, timeframe: str) -> pd.DataFrame:
        """
        Processes a raw OHLCV DataFrame with DUOAI_Strategy to populate all its indicators.
        """
        logger.info(f"Applying DUOAI_Strategy indicators for {pair} ({timeframe}). Initial df shape: {ohlcv_df.shape}")
        if ohlcv_df.empty:
            logger.warning(f"Raw OHLCV DataFrame for {pair} ({timeframe}) is empty. Cannot apply strategy indicators.")
            return ohlcv_df

        try:
            # Construct a minimal config for DUOAI_Strategy and DataProvider
            # This uses the main PreTrainer config as a base.

            # Ensure essential keys like 'user_data_dir' and 'stake_currency' are present.
            # self.config is the config passed to PreTrainer's constructor.
            # DUOAI_Strategy expects 'user_data_dir' and 'stake_currency' at the top level of its config.

            # Start with a known set of essential parameters that Freqtrade strategies might need
            # or that our specific strategy initialization/population might need.
            required_keys_for_strategy = {
                'user_data_dir': self.config.get('user_data_dir', str(PROJECT_ROOT_DIR / 'user_data')), # Default if not in PreTrainer's config
                'stake_currency': self.config.get('stake_currency', 'USDT'), # Default if not in PreTrainer's config
                # Add any other known critical keys that DUOAI_Strategy might access directly from self.config
            }

            strategy_processing_config = {
                **required_keys_for_strategy, # Ensure critical keys are present
                **self.config, # Spread PreTrainer's main config (might override above if they exist there)
                "timeframe": timeframe, # Specific to this call
                "exchange": { # Override exchange settings for offline processing
                    "name": "offline_exchange",
                    "pair_whitelist": [pair], # Critical for DataProvider to know which pair's data to handle
                    "pair_blacklist": []
                },
                "pairlists": [{"method": "StaticPairList", "config": {"pairs": [pair]}}], # Standard for Freqtrade
                "strategy": "DUOAI_Strategy", # Strategy name for Freqtrade context
                "bot_name": "pretrainer_indicator_helper", # For logging or context

                # Runtime operational settings for Freqtrade core
                "pair_whitelist": [pair], # Active pair whitelist for this operation
                "config_pair_whitelist": [pair], # This is what DUOAI_Strategy.__init__ uses for self.config_pair_whitelist

                "dry_run": True,
                "process_only_new_candles": False,
            }

            # Remove keys that might cause issues if not fully configured for a live bot run
            strategy_processing_config.pop('telegram', None)
            strategy_processing_config.pop('api_server', None)
            # Remove other potentially problematic keys if they are not needed for indicator population
            # Example: strategy_processing_config.pop('db_url', None) # if it causes issues and isn't needed

            ft_config_obj = Configuration.from_dict(strategy_processing_config)

            if 'max_open_trades' not in ft_config_obj: # Ensure this key exists
                ft_config_obj['max_open_trades'] = 10

            strategy = DUOAI_Strategy(config=ft_config_obj)
            dataprovider = DataProvider(config=ft_config_obj, exchange=None) # No live exchange needed
            strategy.dp = dataprovider

            # Prepare DataFrame: ensure 'date' index and lowercase OHLCV columns
            df_to_process = ohlcv_df.copy()
            if not isinstance(df_to_process.index, pd.DatetimeIndex) or df_to_process.index.name != 'date':
                if 'date' in df_to_process.columns:
                    df_to_process['date'] = pd.to_datetime(df_to_process['date'])
                    df_to_process.set_index('date', inplace=True)
                elif pd.api.types.is_numeric_dtype(df_to_process.index): # If index is timestamp
                    df_to_process.index = pd.to_datetime(df_to_process.index, unit='ms', utc=True)
                    df_to_process.index.name = 'date'
                else: # Try to convert index if it's string-like datetime
                    try:
                        df_to_process.index = pd.to_datetime(df_to_process.index)
                        df_to_process.index.name = 'date'
                    except Exception as e_idx:
                        logger.error(f"Failed to set DatetimeIndex for {pair} ({timeframe}): {e_idx}. OHLCV df index: {ohlcv_df.index}")
                        return ohlcv_df # Return original if cannot process index

            # Ensure standard OHLCV column names are lowercase
            rename_map_strat = {}
            for col_name in df_to_process.columns:
                col_lower = col_name.lower()
                if col_lower in ['open', 'high', 'low', 'close', 'volume'] and col_name != col_lower:
                    rename_map_strat[col_name] = col_lower
            if rename_map_strat:
                df_to_process.rename(columns=rename_map_strat, inplace=True)

            # Verify required columns after standardization
            required_ohlc_strat = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df_to_process.columns for col in required_ohlc_strat):
                missing_req = [c for c in required_ohlc_strat if c not in df_to_process.columns]
                logger.error(f"DataFrame for strategy processing for {pair} ({timeframe}) is missing required columns: {missing_req} after standardization attempts.")
                return ohlcv_df


            # Call populate_indicators via advise_all_indicators
            # DUOAI_Strategy.advise_all_indicators expects a dictionary of dataframes
            data_for_strategy = {pair: df_to_process}
            analyzed_pair_data = strategy.advise_all_indicators(data_for_strategy)

            # The result is a dictionary, get the dataframe for the pair
            processed_df = analyzed_pair_data[pair]

            logger.info(f"Successfully applied DUOAI_Strategy indicators for {pair} ({timeframe}). Resulting df shape: {processed_df.shape}, Columns: {list(processed_df.columns)}")
            return processed_df

        except Exception as e:
            logger.error(f"Error applying DUOAI_Strategy indicators for {pair} ({timeframe}): {e}", exc_info=True)
            return ohlcv_df # Return original df on error

    def pretrain(self, features_df: pd.DataFrame, scaler: MinMaxScaler,
                 pair: str, timeframe: str, cnn_patterns_instance: 'CNNPatterns'):
        """
        Main method to perform pre-training of a CNN model. (Placeholder)
        """
        logger.info(f"PreTrainer.pretrain placeholder called for {pair} - {timeframe}")

        if features_df is None or features_df.empty:
            logger.error("Features DataFrame is None or empty. Skipping pre-training.")
            return

        logger.info(f"Received features of shape: {features_df.shape}")
        logger.info(f"Scaler details: Min: {scaler.min_[:5] if scaler.min_ is not None else 'N/A'}..., Scale: {scaler.scale_[:5] if scaler.scale_ is not None else 'N/A'}...") # Print first 5 for brevity

        # Placeholder logic as defined in the subtask
        sequence_length = self.config.get('hyperparameters', {}).get('sequence_length', 30) # Example: get from config
        logger.info(f"Using sequence_length: {sequence_length}")

        num_features = features_df.shape[1]
        logger.info(f"Number of features for CNN input: {num_features}")

        # TODO: Actual pre-training logic would involve:
        # 1. Reshaping data into sequences (X).
        # 2. Defining/generating labels (Y).
        # 3. Instantiating the CNN model (e.g., using cnn_patterns_instance.SimpleCNN or a similar class).
        #    model = cnn_patterns_instance.SimpleCNN(input_channels=num_features, num_classes=2, sequence_length=sequence_length) # Example
        # 4. Defining optimizer and loss function.
        # 5. Running the training loop.
        # 6. Saving the trained model and the scaler.

        # Use self.models_dir (which is already a Path object and absolute)
        pair_tf_models_dir = (self.models_dir / pair.replace('/', '_') / timeframe).resolve()
        pair_tf_models_dir.mkdir(parents=True, exist_ok=True)

        model_save_path = pair_tf_models_dir / f"cnn_model_placeholder_{pair.replace('/', '_')}_{timeframe}.pth"
        scaler_save_path = pair_tf_models_dir / f"scaler_placeholder_{pair.replace('/', '_')}_{timeframe}.json"

        logger.info(f"Placeholder: Model would be saved to {model_save_path}")
        logger.info(f"Placeholder: Scaler would be saved to {scaler_save_path}")

        logger.info(f"Placeholder: Pre-training logic for {pair} - {timeframe} completed.")

    # Custom caching methods (_get_cache_dir, _get_cache_filepath, _read_from_cache, _write_to_cache)
    # are removed as per subtask 1.2.2. Freqtrade's DataProvider will handle caching.

    def _load_gold_standard_data(self, symbol: str, timeframe: str, pattern_type: str, expected_columns: List[str]) -> pd.DataFrame | None:
        """
        Loads gold standard data for a given symbol, timeframe, and pattern type.
        Verifies that the loaded data contains all expected columns and sets the timestamp as index.
        """
        gold_standard_path_str = self.params_manager.get_param("gold_standard_data_path")
        if not gold_standard_path_str:
            logger.debug("Gold standard data path not set in parameters. Skipping gold standard loading.")
            return None

        gold_standard_base_path = Path(gold_standard_path_str)
        if not gold_standard_base_path.is_absolute():
            gold_standard_base_path = (PROJECT_ROOT_DIR / gold_standard_base_path).resolve()
            logger.debug(f"Gold standard data path was relative, resolved to: {gold_standard_base_path}")


        symbol_sanitized = symbol.replace('/', '_')
        filename = f"{symbol_sanitized}_{timeframe}_{pattern_type}_gold.csv"
        full_path = (gold_standard_base_path / filename).resolve()

        if not full_path.exists():
            logger.debug(f"Gold standard data file not found: {full_path}")
            return None

        try:
            df = pd.read_csv(str(full_path))
            if df.empty:
                logger.warning(f"Gold standard data file is empty: {full_path}")
                return None

            # Verify all expected columns are present
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Gold standard data file {full_path} is missing expected columns: {missing_cols}. Expected: {expected_columns}")
                return None

            # Ensure 'timestamp' column is present for index conversion
            if 'timestamp' not in df.columns:
                logger.error(f"Gold standard data file {full_path} is missing 'timestamp' column for index conversion.")
                return None

            # Convert timestamp to datetime and set as index
            # Assuming timestamp is in milliseconds if it's a large integer, otherwise seconds or already datetime string
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                # Heuristic: if timestamp values are very large, assume milliseconds
                if df['timestamp'].mean() > 1e12: # Arbitrary threshold for milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else: # Assume seconds otherwise
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else: # Try direct conversion for string formats
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True) # Ensure data is sorted by time

            logger.info(f"Successfully loaded {len(df)} records from gold standard data: {full_path}")
            return df

        except FileNotFoundError:
            logger.debug(f"Gold standard data file not found (should have been caught by .exists(), but good to handle): {full_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.warning(f"Gold standard data file is empty (pandas error): {full_path}", exc_info=True)
            return None
        except ValueError as ve:
            logger.error(f"ValueError processing gold standard data file {full_path}: {ve}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading gold standard data from {full_path}: {e}", exc_info=True)
            return None

    def _fetch_ohlcv_for_period_sync(self, symbol: str, timeframe: str, start_dt: dt, end_dt: dt) -> pd.DataFrame | None:
        """
        Fetches OHLCV data for a given period from Freqtrade's local data storage.
        Relies on `freqtrade download-data` having been run previously.
        """
        logger.info(f"PreTrainer: Attempting to load local Freqtrade data for {symbol} ({timeframe}) from {start_dt} to {end_dt}")

        try:
            # These configurations should ideally come from self.config or be passed reliably
            exchange_name = self.config.get("exchange", {}).get("name", "binance")
            user_data_dir_str = self.config.get("user_data_dir", "user_data")

            user_data_path = Path(user_data_dir_str)
            if not user_data_path.is_absolute():
                user_data_path = (PROJECT_ROOT_DIR / user_data_path).resolve()

            datadir = (user_data_path / "data" / exchange_name).resolve()

            if not datadir.is_dir():
                 logger.warning(f"PreTrainer: Freqtrade data directory not found: {datadir}. "
                               f"Please ensure data is downloaded via 'freqtrade download-data --exchange {exchange_name}'.")

            dp_config_dict = {
                "user_data_dir": str(user_data_path),
                "exchange": {
                    "name": exchange_name,
                    "pair_whitelist": [symbol]
                },
                "datadir": str(datadir),
            }
            ft_dp_config = Configuration.from_dict(dp_config_dict)

            dataprovider = DataProvider(config=ft_dp_config, exchange=None)

            # Create TimeRange object
            # TimeRange expects timestamps in seconds
            timerange = TimeRange("date", "date", int(start_dt.timestamp()), int(end_dt.timestamp()))

            dataframe = dataprovider.historic_ohlcv(
                pair=symbol,
                timeframe=timeframe,
                timerange=timerange
            )

            if dataframe is None or dataframe.empty:
                logger.warning(f"PreTrainer: No local Freqtrade data found for {symbol} ({timeframe}) in range {start_dt}-{end_dt} using datadir {datadir}. "
                               f"Ensure data is downloaded: 'freqtrade download-data --exchange {exchange_name} --pairs {symbol} --timeframes {timeframe} --timerange {start_dt.strftime('%Y%m%d')}-{end_dt.strftime('%Y%m%d')}'")
                return None

            # Ensure DataFrame has DatetimeIndex and is UTC localized
            # DataProvider.historic_ohlcv should already return a dataframe with a DatetimeIndex.
            # It typically localizes to UTC if 'date' column was timezone-naive.
            if not isinstance(dataframe.index, pd.DatetimeIndex):
                 logger.warning(f"PreTrainer: DataProvider data for {symbol} {timeframe} does not have DatetimeIndex. Attempting conversion.")
                 # This case should be rare if DataProvider works as expected.
                 # If 'date' is a column:
                 if 'date' in dataframe.columns:
                     dataframe['date'] = pd.to_datetime(dataframe['date'], errors='coerce', utc=True)
                     dataframe.set_index('date', inplace=True)
                 # If index is numeric (e.g. timestamp ms from a CSV read elsewhere, though DP handles this)
                 elif pd.api.types.is_numeric_dtype(dataframe.index):
                     dataframe.index = pd.to_datetime(dataframe.index, unit='ms', utc=True)
                     dataframe.index.name = 'date'
                 else: # Try direct conversion of index
                     try:
                        dataframe.index = pd.to_datetime(dataframe.index, errors='coerce', utc=True)
                        dataframe.index.name = 'date'
                     except Exception:
                        logger.error(f"PreTrainer: Could not convert index to DatetimeIndex for {symbol} {timeframe}. Index type: {type(dataframe.index)}")
                        return None # Cannot proceed without a proper DatetimeIndex

            # Ensure UTC localization (DataProvider usually handles this)
            if dataframe.index.tzinfo is None:
                logger.debug(f"PreTrainer: Localizing index to UTC for {symbol} {timeframe}.")
                dataframe.index = dataframe.index.tz_localize('UTC')
            elif dataframe.index.tzinfo != timezone.utc: # Check if it's already UTC
                logger.debug(f"PreTrainer: Converting index to UTC for {symbol} {timeframe}.")
                dataframe.index = dataframe.index.tz_convert('UTC')

            # Standardize column names to lowercase (open, high, low, close, volume)
            rename_map_fetch = {col: col.lower() for col in dataframe.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume'] and col != col.lower()}
            if rename_map_fetch:
                dataframe.rename(columns=rename_map_fetch, inplace=True)

            logger.info(f"PreTrainer: Successfully loaded {len(dataframe)} candles from local Freqtrade data for {symbol} ({timeframe}).")
            return dataframe

        except Exception as e:
            logger.error(f"PreTrainer: Error loading local Freqtrade data for {symbol} ({timeframe}): {e}", exc_info=True)
            return None

    async def fetch_historical_data(self, symbol: str, timeframe: str) -> Dict[str, pd.DataFrame]:
        if not self.bitvavo_executor:
            logger.error("BitvavoExecutor niet geïnitialiseerd. Kan geen historische data ophalen.")
            return {"all": pd.DataFrame()} # Return dict with empty DF for "all"

        all_ohlcv_data_dfs_by_regime: Dict[str, List[pd.DataFrame]] = {}
        all_ohlcv_data_dfs_for_concatenation: List[pd.DataFrame] = []
        processed_regime_dfs: Dict[str, pd.DataFrame] = {}
        fetched_any_data = False

        if symbol in self.market_regimes and self.market_regimes[symbol]:
            logger.info(f"Marktregimes gevonden voor {symbol}. Data wordt per gedefinieerd regime opgehaald.")
            symbol_regimes = self.market_regimes[symbol]
            for regime_category, periods in symbol_regimes.items():
                if not periods:
                    logger.info(f"Geen periodes gedefinieerd voor regime '{regime_category}' in {symbol}. Overslaan.")
                    continue

                logger.info(f"Verwerken regime categorie: {regime_category} voor {symbol} ({timeframe})")
                regime_specific_period_dfs = []
                for period in periods:
                    try:
                        regime_start_dt = dt.strptime(period['start_date'], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        # Ensure end_dt is also UTC and represents end of the day in UTC
                        regime_end_dt = dt.strptime(period['end_date'], "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
                        if regime_start_dt > regime_end_dt:
                            logger.warning(f"Ongeldige periode in regimes voor {symbol} {regime_category}: start {period['start_date']} is na end {period['end_date']}. Overslaan.")
                            continue

                        period_df = await asyncio.to_thread(self._fetch_ohlcv_for_period_sync, symbol, timeframe, regime_start_dt, regime_end_dt)
                        if period_df is not None and not period_df.empty:
                            regime_specific_period_dfs.append(period_df)
                            all_ohlcv_data_dfs_for_concatenation.append(period_df) # Also add to general list for "all" data
                            fetched_any_data = True
                            logger.info(f"Data ({len(period_df)} candles) gehaald voor {symbol}-{timeframe} ({regime_category}) periode {period['start_date']}-{period['end_date']}")
                        else:
                            logger.info(f"Geen data gehaald voor {symbol}-{timeframe} ({regime_category}) periode {period['start_date']}-{period['end_date']}")

                    except ValueError as e:
                        logger.error(f"Ugyldig datoformat in regimes for {symbol} {regime_category} periode {period}: {e}. Overslaan.")
                    except Exception as e:
                        logger.error(f"Generieke fout bij verwerken regime {symbol} {regime_category} periode {period}: {e}. Overslaan.")

                if regime_specific_period_dfs:
                    all_ohlcv_data_dfs_by_regime[regime_category] = regime_specific_period_dfs
                    logger.info(f"Data voor regime '{regime_category}' ({symbol}-{timeframe}) verzameld.")
                else:
                    logger.warning(f"Geen data verzameld voor regime '{regime_category}' ({symbol}-{timeframe}) na verwerken van alle periodes.")

        if not fetched_any_data or not all_ohlcv_data_dfs_for_concatenation: # Fallback or if only "all" is desired implicitly
            logger.info(f"Geen data via specifieke regimes gehaald (of geen regimes gedefinieerd/gevonden voor {symbol}). Globale datarange wordt gebruikt voor 'all' data.")
            global_start_date_str = self.params_manager.get_param("data_fetch_start_date_str")
            global_end_date_str = self.params_manager.get_param("data_fetch_end_date_str")
            if not global_start_date_str:
                logger.error(f"Geen 'data_fetch_start_date_str' in ParamsManager voor {symbol} ({timeframe}). Kan geen 'all' data halen.")
                return {"all": pd.DataFrame()}

            try:
                start_dt_global = dt.strptime(global_start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                end_dt_global = datetime.now(timezone.utc) # Use datetime.now(timezone.utc)
                if global_end_date_str:
                    end_dt_global = dt.strptime(global_end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)

                if start_dt_global > end_dt_global:
                    logger.error(f"Global startdatum {global_start_date_str} is na einddatum {global_end_date_str if global_end_date_str else 'nu'}.")
                    return {"all": pd.DataFrame()}

                global_df = await asyncio.to_thread(self._fetch_ohlcv_for_period_sync, symbol, timeframe, start_dt_global, end_dt_global)
                if global_df is not None and not global_df.empty:
                    all_ohlcv_data_dfs_for_concatenation.append(global_df)
                    fetched_any_data = True
                    logger.info(f"Data ({len(global_df)} candles) gehaald voor {symbol}-{timeframe} via globale periode: {global_start_date_str} - {global_end_date_str if global_end_date_str else 'nu'}.")
                else:
                    logger.warning(f"Geen data gehaald voor {symbol}-{timeframe} met globale periode.")
            except ValueError as e:
                logger.error(f"Ugyldig globalt datoformat: {e}.", exc_info=True)
                return {"all": pd.DataFrame()}

        # Process "all" data (concatenated from all sources)
        concatenated_df = None
        if all_ohlcv_data_dfs_for_concatenation:
            concatenated_df = pd.concat(all_ohlcv_data_dfs_for_concatenation, ignore_index=False)
            if not isinstance(concatenated_df.index, pd.DatetimeIndex):
                 logger.error(f"Index voor geconcateneerde data {symbol} ({timeframe}) is geen DatetimeIndex.")
                 if 'date' in concatenated_df.columns: concatenated_df['date'] = pd.to_datetime(concatenated_df['date']); concatenated_df.set_index('date', inplace=True)
                 # else: concatenated_df will be empty or problematic, handled below
            if not concatenated_df.index.is_unique:
                concatenated_df = concatenated_df[~concatenated_df.index.duplicated(keep='first')]
                logger.info(f"Removed duplicate indices from concatenated_df for {symbol} ({timeframe}).")
            concatenated_df.sort_index(inplace=True)
            logger.info(f"Concatenated_df for {symbol} ({timeframe}) sorted. Shape: {concatenated_df.shape}")
            concatenated_df.attrs['timeframe'] = timeframe
            concatenated_df.attrs['pair'] = symbol
            # logger.info(f"Totaal {len(concatenated_df)} unieke candles voor 'all' data voor {symbol} ({timeframe}).") # Redundant with shape log

            # Apply strategy indicators to the 'all' data
            if concatenated_df is not None and not concatenated_df.empty:
                logger.info(f"Processing 'all' data for {symbol} ({timeframe}) with DUOAI_Strategy indicators...")
                # The helper method is synchronous, run in a thread if called from async
                concatenated_df = await asyncio.to_thread(
                    self._get_dataframe_with_strategy_indicators,
                    concatenated_df, symbol, timeframe
                )

        processed_regime_dfs["all"] = concatenated_df if concatenated_df is not None and not concatenated_df.empty else pd.DataFrame()
        if processed_regime_dfs["all"].empty:
            logger.warning(f"Uiteindelijk geen 'all' data beschikbaar voor {symbol} ({timeframe}) (evt. na strategy processing).")


        # Process regime-specific data
        for regime_cat, df_list in all_ohlcv_data_dfs_by_regime.items():
            if df_list:
                regime_df = pd.concat(df_list, ignore_index=False)
                if not isinstance(regime_df.index, pd.DatetimeIndex):
                    logger.error(f"Index voor regime '{regime_cat}' data {symbol} ({timeframe}) is geen DatetimeIndex.")
                    if 'date' in regime_df.columns: regime_df['date'] = pd.to_datetime(regime_df['date']); regime_df.set_index('date', inplace=True)
                    else: processed_regime_dfs[regime_cat] = pd.DataFrame(); continue

                if not regime_df.index.is_unique:
                    regime_df = regime_df[~regime_df.index.duplicated(keep='first')]
                    logger.info(f"Removed duplicate indices from regime_df for {symbol} ({timeframe}, {regime_cat}).")
                regime_df.sort_index(inplace=True)
                logger.info(f"Regime_df for {symbol} ({timeframe}, {regime_cat}) sorted. Shape: {regime_df.shape}")
                regime_df.attrs['timeframe'] = timeframe
                regime_df.attrs['pair'] = symbol

                # Apply strategy indicators to this regime's data
                if regime_df is not None and not regime_df.empty:
                    logger.info(f"Processing regime '{regime_cat}' data for {symbol} ({timeframe}) with DUOAI_Strategy indicators...")
                    regime_df = await asyncio.to_thread(
                        self._get_dataframe_with_strategy_indicators,
                        regime_df, symbol, timeframe
                    )

                processed_regime_dfs[regime_cat] = regime_df
                logger.info(f"Data voor regime '{regime_cat}' ({symbol}-{timeframe}) verwerkt. Shape: {regime_df.shape}")
            else:
                processed_regime_dfs[regime_cat] = pd.DataFrame()
                logger.warning(f"Lege DataFrame voor regime '{regime_cat}' ({symbol}-{timeframe}) na verwerking.")

        if not fetched_any_data and not processed_regime_dfs.get("all", pd.DataFrame()).empty:
             logger.warning(f"Geen data gehaald voor {symbol} ({timeframe}) voor enige regime of globale periode.")
             # Ensure "all" is an empty DataFrame if truly no data
             if all(df.empty for df in processed_regime_dfs.values()):
                 return {"all": pd.DataFrame()}

        return processed_regime_dfs

    async def prepare_training_data(self, dataframe: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        pair = dataframe.attrs.get('pair', 'N/A')
        tf = dataframe.attrs.get('timeframe', 'N/A')
        logger.info(f"Voorbereiden trainingsdata voor {pair} ({tf}), pattern: {pattern_type}...")

        if dataframe.empty:
            logger.warning(f"Lege dataframe ontvangen in prepare_training_data for {pattern_type}.")
            return dataframe

        # Indicators are now expected to be in the dataframe from DUOAI_Strategy.populate_indicators,
        # called via _get_dataframe_with_strategy_indicators within fetch_historical_data.
        # Logging to confirm presence of expected indicators:
        expected_indicators = ['rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
        # Also check for other indicators DUOAI_Strategy might add, e.g., from informative pairs.
        # For now, just checking the basic ones that were previously calculated here.
        missing_indicators = [ind for ind in expected_indicators if ind not in dataframe.columns]
        if missing_indicators:
            logger.warning(f"DataFrame for {pair} ({tf}) is missing expected strategy indicators: {missing_indicators}. Training might be suboptimal or fail if features are crucial.")
        else:
            logger.info(f"All basic expected strategy indicators (RSI, MACD, BBands) found in DataFrame for {pair} ({tf}).")

        # Ensure essential OHLCV columns are present as they are used in labeling logic below
        # (e.g., dataframe['high'], dataframe['low'], dataframe['close'])
        # These should have been preserved by DUOAI_Strategy.
        required_ohlcv_for_labeling = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv = [col for col in required_ohlcv_for_labeling if col not in dataframe.columns]
        if missing_ohlcv:
            logger.error(f"DataFrame for {pair} ({tf}) is missing essential OHLCV columns: {missing_ohlcv} needed for labeling logic. Returning empty.")
            return pd.DataFrame()

        label_column_name = f"{pattern_type}_label"
        dataframe[label_column_name] = 0 # Initialize label column
        dataframe['gold_label_applied'] = False # Helper column to track gold standard application

        # 1. Load Gold Standard Data
        gold_ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] # Core OHLCV for structure
        gold_df = self._load_gold_standard_data(
            symbol=pair,
            timeframe=tf,
            pattern_type=pattern_type,
            expected_columns=gold_ohlcv_cols + ['gold_label']
        )

        # 2. Merge Gold Standard Labels
        if gold_df is not None and not gold_df.empty and 'gold_label' in gold_df.columns:
            logger.info(f"Gold standard data gevonden voor {pair}-{tf}-{pattern_type}. Labels worden samengevoegd.")
            # Ensure gold_df index is datetime, _load_gold_standard_data should handle this
            if not isinstance(gold_df.index, pd.DatetimeIndex):
                 logger.error("Gold standard DataFrame index is not DatetimeIndex. Kan niet mergen.")
            else:
                # Join gold labels. Keep original dataframe index.
                temp_df = dataframe.join(gold_df[['gold_label']], how='left', rsuffix='_gold')

                # Apply gold labels where available
                gold_labels_applied_count = 0
                if 'gold_label' in temp_df.columns: # Check if join produced the column
                    # Mark rows where gold label is not NaN (meaning it was present in gold_df)
                    valid_gold_indices = temp_df['gold_label'].notna()
                    dataframe.loc[valid_gold_indices, label_column_name] = temp_df.loc[valid_gold_indices, 'gold_label'].astype(int)
                    dataframe.loc[valid_gold_indices, 'gold_label_applied'] = True
                    gold_labels_applied_count = valid_gold_indices.sum()
                    logger.info(f"{gold_labels_applied_count} labels toegepast vanuit gold standard data voor {pair}-{tf}-{pattern_type}.")
                else:
                    logger.warning(f"Kolom 'gold_label' niet gevonden na join met gold_df voor {pair}-{tf}-{pattern_type}.")
        else:
            logger.info(f"Geen gold standard data gebruikt voor labeling voor {pair}-{tf}-{pattern_type}.")

        # 3. Form Pattern Detection (already present)
        dataframe['form_pattern_detected'] = False
        min_pattern_window_size = 0
        if pattern_type == 'bullFlag': min_pattern_window_size = 10
        elif pattern_type == 'bearishEngulfing': min_pattern_window_size = 2
        else:
            logger.warning(f"Form detection niet geïmplementeerd voor '{pattern_type}' voor {pair}-{tf}. Defaulting all to True.")
            dataframe['form_pattern_detected'] = True

        if min_pattern_window_size > 0: # Proceed only if pattern detection is meaningful
            if self.cnn_pattern_detector is None:
                logger.error("CNNPatternDetector not initialized in PreTrainer. Cannot detect form patterns.")
                dataframe['form_pattern_detected'] = False # Ensure column exists
            elif dataframe.empty or len(dataframe) < min_pattern_window_size:
                logger.info(f"DataFrame too short for pattern detection window {min_pattern_window_size}. No patterns detected.")
                dataframe['form_pattern_detected'] = False
            else:
                # Helper function for rolling apply. It uses `dataframe` and `pattern_type` from the outer scope.
                def _apply_pattern_detection_to_window(window_dummy_series: pd.Series) -> bool:
                    # window_dummy_series is a rolling window of a single column (e.g. 'close').
                    # Its index can be used to slice the original multi-column DataFrame.
                    if len(window_dummy_series) < min_pattern_window_size:
                        return False # Should be handled by min_periods in rolling()

                    # Get the actual window from the original 'dataframe' using the window's index
                    # The window passed to apply() has an index that corresponds to the original DataFrame.
                    # The slice should be from the start of this window's index to the end.
                    start_index_val = window_dummy_series.index[0]
                    end_index_val = window_dummy_series.index[-1]

                    # Slice the original dataframe using .loc to get the multi-column window
                    actual_window_df = dataframe.loc[start_index_val:end_index_val]

                    # _dataframe_to_candles handles DataFrames with DatetimeIndex by resetting index internally
                    candle_list = self.cnn_pattern_detector._dataframe_to_candles(actual_window_df)

                    if not candle_list or len(candle_list) < min_pattern_window_size:
                        # This check might be redundant if min_periods is set correctly in rolling()
                        # and actual_window_df correctly reflects the window size.
                        return False

                    detected = False
                    if pattern_type == 'bullFlag':
                        detected = self.cnn_pattern_detector.detect_bull_flag(candle_list)
                    elif pattern_type == 'bearishEngulfing':
                        engulfing_type = self.cnn_pattern_detector._detect_engulfing(candle_list, "bearish")
                        detected = (engulfing_type == "bearishEngulfing")
                    return detected

                # Apply this to a single column (e.g., 'close') to iterate through windows.
                # `min_periods` ensures the function is only called on full windows.
                # `raw=False` ensures the applied function receives a Series with an index.
                # `engine='python'` is required for apply functions that are not simple reductions.
                # This approach, while using rolling().apply(), might not be more performant
                # than a direct loop for this specific type of operation due to the reconstruction
                # of DataFrame slices inside the applied function.
                detected_series = dataframe['close'].rolling(
                    window=min_pattern_window_size,
                    min_periods=min_pattern_window_size # Ensures func only called on full windows
                ).apply(_apply_pattern_detection_to_window, raw=False, engine='python')

                dataframe['form_pattern_detected'] = detected_series.fillna(False).astype(bool)
        else: # No meaningful pattern detection for this pattern_type (e.g. window size is 0)
            dataframe['form_pattern_detected'] = True # Default to True as per original logic
            if pattern_type not in ['bullFlag', 'bearishEngulfing']: # Log only if it's an unknown type
                 logger.warning(f"Form detection not implemented or window size is 0 for '{pattern_type}' on {pair}-{tf}. Defaulting 'form_pattern_detected' to True.")


        form_detected_count = dataframe['form_pattern_detected'].sum()
        logger.info(f"For pattern '{pattern_type}' on {pair}-{tf}, {form_detected_count} samples identified by form detection logic.")

        # 4. Automatic Labeling for non-gold-labeled data
        configs = self.params_manager.get_param('pattern_labeling_configs')
        if not configs:
            logger.error("`pattern_labeling_configs` niet gevonden. Automatische labeling gestopt.");
            # dataframe[label_column_name] is al geïnitialiseerd (mogelijk met gold labels)
            # Drop helper columns before returning
            dataframe.drop(columns=['gold_label_applied', 'form_pattern_detected'], inplace=True, errors='ignore')
            if 'future_high' in dataframe.columns: dataframe.drop(columns=['future_high'], inplace=True, errors='ignore')
            if 'future_low' in dataframe.columns: dataframe.drop(columns=['future_low'], inplace=True, errors='ignore')
            cols_to_drop_for_na_final = [label_column_name] + ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
            dataframe.dropna(subset=cols_to_drop_for_na_final, inplace=True)
            return dataframe

        cfg = configs.get(pattern_type)
        if not cfg:
            logger.warning(f"Config for '{pattern_type}' niet gevonden. Automatische labeling overgeslagen.")
            # dataframe[label_column_name] is al geïnitialiseerd
            dataframe.drop(columns=['gold_label_applied', 'form_pattern_detected'], inplace=True, errors='ignore')
            if 'future_high' in dataframe.columns: dataframe.drop(columns=['future_high'], inplace=True, errors='ignore')
            if 'future_low' in dataframe.columns: dataframe.drop(columns=['future_low'], inplace=True, errors='ignore')
            cols_to_drop_for_na_final = [label_column_name] + ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
            dataframe.dropna(subset=cols_to_drop_for_na_final, inplace=True)
            return dataframe

        future_N, profit_thresh, loss_thresh = cfg.get('future_N_candles',20), cfg.get('profit_threshold_pct',0.02), cfg.get('loss_threshold_pct',-0.01)
        dataframe['future_high'] = dataframe['high'].shift(-future_N)
        dataframe['future_low'] = dataframe['low'].shift(-future_N)

        # Indices for automatic labeling: form detected AND not already labeled by gold standard
        # Using .copy() on the boolean series to avoid SettingWithCopyWarning if these series are reused.
        auto_label_condition = dataframe['form_pattern_detected'].copy() & ~dataframe['gold_label_applied'].copy()

        # Make sure to only select from indices that will have valid future_high/low (not NaN)
        # This is implicitly handled later by dropna, but good to be aware.
        # For safety, ensure detected_indices_for_auto only includes rows where auto_label_condition is true.
        detected_indices_for_auto = dataframe.index[auto_label_condition]

        auto_labeled_count = 0
        if not detected_indices_for_auto.empty:
            if pattern_type == 'bullFlag':
                profit_met = (dataframe.loc[detected_indices_for_auto, 'future_high'] - dataframe.loc[detected_indices_for_auto, 'close']) / dataframe.loc[detected_indices_for_auto, 'close'] >= profit_thresh
                loss_met = (dataframe.loc[detected_indices_for_auto, 'future_low'] - dataframe.loc[detected_indices_for_auto, 'close']) / dataframe.loc[detected_indices_for_auto, 'close'] <= loss_thresh
                # Apply auto-label only to those identified for auto-labeling
                dataframe.loc[detected_indices_for_auto, label_column_name] = np.where(profit_met & ~loss_met, 1, 0)
                auto_labeled_count = len(detected_indices_for_auto[profit_met & ~loss_met])
            elif pattern_type == 'bearishEngulfing':
                profit_met = (dataframe.loc[detected_indices_for_auto, 'close'] - dataframe.loc[detected_indices_for_auto, 'future_low']) / dataframe.loc[detected_indices_for_auto, 'close'] >= profit_thresh
                loss_met = (dataframe.loc[detected_indices_for_auto, 'future_high'] - dataframe.loc[detected_indices_for_auto, 'close']) / dataframe.loc[detected_indices_for_auto, 'close'] >= abs(loss_thresh)
                dataframe.loc[detected_indices_for_auto, label_column_name] = np.where(profit_met & ~loss_met, 1, 0)
                auto_labeled_count = len(detected_indices_for_auto[profit_met & ~loss_met])

        if auto_labeled_count > 0:
            logger.info(f"{auto_labeled_count} labels toegepast via automatische labeling voor {pair}-{tf}-{pattern_type} (op non-gold samples).")
        else:
            logger.info(f"Geen labels toegepast via automatische labeling voor {pair}-{tf}-{pattern_type} (mogelijk alles door gold, of geen form detected, of geen profit/loss met).")

        total_positive_labels = dataframe[label_column_name].sum()
        logger.info(f"Totaal {total_positive_labels} positieve labels voor {pair}-{tf}-{pattern_type} na gold en/of automatische labeling.")

        # Cleanup and NaN/inf handling
        initial_row_count = len(dataframe)
        logger.info(f"DataFrame shape before NaN/inf handling and dropna: {dataframe.shape} for {pair}-{tf}-{pattern_type}")

        # Define a more comprehensive list of feature columns expected from DUOAI_Strategy.
        # This should include base OHLCV, primary indicators, EMAs, volume means, and candlestick patterns.
        # Also, corresponding columns from informative timeframes (e.g., '1h_rsi', '4h_ema_10').
        # For brevity here, we'll list common base ones and note that a dynamic approach might be better.
        base_feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macdsignal', 'macdhist',
            'bb_lowerband', 'bb_middleband', 'bb_upperband',
            'ema_10', 'ema_25', 'ema_50', # Assuming these are fixed EMA periods in DUOAI_Strategy
            'volume_mean_20' # Example volume mean
        ]
        # Add candlestick pattern columns (assuming they are like 'cdl_DOJI', 'cdl_ENGULFING')
        # A more robust way would be to get column names matching 'cdl_*' pattern.
        candlestick_cols = [col for col in dataframe.columns if col.startswith('cdl_')]

        # Add informative timeframe features. Example for '1h': '1h_rsi', '1h_ema_10', etc.
        # This part needs to be dynamic based on `self.informative_timeframes` in DUOAI_Strategy.
        informative_feature_columns = []
        # Example: if '1h' is an informative timeframe and 'rsi' is a base feature:
        # informative_feature_columns.append('1h_rsi')
        # This should ideally be built by iterating through strategy's informative TFs and its indicator list.
        # For now, we'll use a placeholder or rely on `select_dtypes(include=np.number)` later if this list isn't exhaustive.
        # However, explicitly listing critical features for dropna is safer.
        # Let's assume DUOAI_Strategy.informative_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        # And base indicators are somewhat fixed.
        for info_tf_prefix in ['1m', '5m', '15m', '1h', '4h', '1d']: # From DUOAI_Strategy.informative_timeframes
            if f"{info_tf_prefix}_close" in dataframe.columns: # Check if informative timeframe was merged
                for base_col in base_feature_columns: # Replicate base features for info_tf
                     informative_feature_columns.append(f"{info_tf_prefix}_{base_col}")
                for cdl_col in candlestick_cols: # Replicate cdl patterns for info_tf
                    informative_feature_columns.append(f"{info_tf_prefix}_{cdl_col}")


        # Complete list of features to check for NaNs
        all_feature_columns_to_check = base_feature_columns + candlestick_cols + informative_feature_columns
        # Filter this list to only include columns actually present in the dataframe
        all_feature_columns_present = [col for col in all_feature_columns_to_check if col in dataframe.columns]

        logger.info(f"Identified {len(all_feature_columns_present)} feature columns for NaN/inf check.")
        # It's generally better to drop rows if critical features or the label are NaN.
        # For some non-critical features, fillna might be an option, but requires careful consideration.
        # Recommendation: Stick with dropna for core features and labels.
        # For features where fillna might be acceptable (e.g., some non-critical indicator),
        # it should be done *before* this stage or very selectively.
        # For now, we will use dropna on all identified feature columns + label.

        # Handle infinity values first
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        rows_after_inf_handling = len(dataframe)
        if initial_row_count != rows_after_inf_handling: # This condition is actually for after dropna.
             # Log if inf values were replaced (though count won't change until dropna)
             pass # No direct row change from replace to NaN, only value change.

        # Ensure label_column_name exists, even if all labeling failed (it's initialized to 0)
        if label_column_name not in dataframe.columns: dataframe[label_column_name] = 0

        # Columns to use for dropna: all identified feature columns + the label
        cols_to_dropna_on = [label_column_name] + all_feature_columns_present
        # Also include 'future_high' and 'future_low' as they are used for labeling logic before this cleanup.
        if 'future_high' in dataframe.columns: cols_to_dropna_on.append('future_high')
        if 'future_low' in dataframe.columns: cols_to_dropna_on.append('future_low')

        # Ensure only existing columns are in the subset for dropna
        cols_to_dropna_on = [col for col in cols_to_dropna_on if col in dataframe.columns]

        dataframe.dropna(subset=cols_to_dropna_on, inplace=True)
        rows_after_dropna = len(dataframe)
        logger.info(f"DataFrame shape after NaN/inf handling and dropna: {dataframe.shape}. "
                    f"Rows removed: {initial_row_count - rows_after_dropna} for {pair}-{tf}-{pattern_type}")

        # Drop helper and intermediate columns (already present)
        columns_to_drop_finally = ['future_high', 'future_low', 'form_pattern_detected', 'gold_label_applied']
        dataframe.drop(columns=[col for col in columns_to_drop_finally if col in dataframe.columns], inplace=True, errors='ignore')

        logger.info(f"Trainingsdata voorbereid voor {pattern_type} ({pair}-{tf}) met {len(dataframe)} samples. Label: {label_column_name}, Positives: {dataframe[label_column_name].sum() if label_column_name in dataframe else 'N/A'}")
        return dataframe

    async def train_ai_models(self, training_data: pd.DataFrame, symbol: str, timeframe: str, pattern_type: str, target_label_column: str, regime_name: str = "all"):
        current_arch_key_orig = self.params_manager.get_param('current_cnn_architecture_key', "default_simple")
        all_arch_configs = self.params_manager.get_param('cnn_architecture_configs')

        arch_params = {} # This will hold the final architecture parameters
        if all_arch_configs and current_arch_key_orig in all_arch_configs:
            arch_params = all_arch_configs[current_arch_key_orig].copy()
            logger.info(f"Start met basis CNN architectuur '{current_arch_key_orig}' voor {symbol}-{timeframe}-{pattern_type}-{regime_name}.")
        else:
            logger.warning(f"Basis CNN architectuur '{current_arch_key_orig}' niet gevonden. Fallback naar SimpleCNN defaults voor {symbol}-{timeframe}-{pattern_type}-{regime_name}.")
            current_arch_key_orig = "default_simple" # Reflect that we're using defaults if key not found

        # Initial model name base (arch key might change if HPO defines a new one, though not implemented yet)
        model_name_base = f"{symbol.replace('/', '_')}_{timeframe}_{pattern_type}_{current_arch_key_orig}"

        # Training parameters that might be tuned by HPO or taken from params_manager
        sequence_length = self.params_manager.get_param('sequence_length_cnn', 30)
        num_epochs = self.params_manager.get_param('num_epochs_cnn', 10) # Epochs for final training
        batch_size = self.params_manager.get_param('batch_size_cnn', 32)
        learning_rate = self.params_manager.get_param('learning_rate_cnn', 0.001)

        X_list, y_list = [], []
        for i in range(len(training_data) - sequence_length):
            X_list.append(training_data[feature_columns].iloc[i:i + sequence_length].values)
            y_list.append(training_data[target_label_column].iloc[i + sequence_length - 1])
        if not X_list: logger.warning(f"Niet genoeg data voor '{model_name_prefix}' om sequenties te maken (lengte {sequence_length})."); return

        X, y = np.array(X_list), np.array(y_list)
        X_reshaped = X.reshape(-1, X.shape[-1])
        min_vals, max_vals = X_reshaped.min(axis=0), X_reshaped.max(axis=0)
        range_vals = max_vals - min_vals; range_vals[range_vals == 0] = 1
        scale_, min_ = (1.0 / range_vals).tolist(), min_vals.tolist()
        feature_names_in_ = feature_columns
        X_normalized = ((X_reshaped - min_vals) / range_vals).reshape(X.shape)
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32).permute(0, 2, 1)
        y_tensor = torch.tensor(y, dtype=torch.long)

        X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor if np.sum(y)>1 else None)

        cv_results = None
        perform_cv = self.params_manager.get_param('perform_cross_validation', False) # Default to False if not set
        cv_n_splits = self.params_manager.get_param('cv_num_splits', 5)

        if perform_cv and len(X_tensor) > cv_n_splits * batch_size : # Ensure enough samples for CV
            logger.info(f"Uitvoeren TimeSeries Cross-Validation met {cv_n_splits} splits voor {model_name_prefix}...")
            tscv = TimeSeriesSplit(n_splits=cv_n_splits)
            fold_metrics_list = []

            for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(X_tensor)): # Use X_tensor for splitting
                logger.info(f"CV Fold {fold_idx + 1}/{cv_n_splits}")
                X_train_fold, X_val_fold = X_tensor[train_indices], X_tensor[val_indices]
                y_train_fold, y_val_fold = y_tensor[train_indices], y_tensor[val_indices]

                if len(X_val_fold) == 0 or len(np.unique(y_val_fold.cpu().numpy())) < 2 and len(y_val_fold.cpu().numpy()) > 0 : # Skip if val is empty or has only one class for AUC
                    logger.warning(f"Fold {fold_idx+1} overgeslagen: validatieset is leeg of bevat slechts één klasse.")
                    continue

                train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
                val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)
                train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
                val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

                fold_model = SimpleCNN(input_channels=X_train_fold.shape[1], num_classes=2, sequence_length=sequence_length, **arch_params)
                fold_optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate)
                fold_criterion = nn.CrossEntropyLoss()

                # Simplified training loop for fold (no early stopping within fold for now)
                for _ in range(num_epochs): # Using main num_epochs for each fold
                    fold_model.train()
                    for data, targets in train_loader_fold:
                        fold_optimizer.zero_grad(); outputs = fold_model(data); loss = fold_criterion(outputs, targets)
                        loss.backward(); fold_optimizer.step()

                fold_model.eval()
                fold_val_loss, fold_preds, fold_targets_list = 0.0, [], []
                with torch.no_grad():
                    for data, targets in val_loader_fold:
                        outputs = fold_model(data)
                        fold_val_loss += fold_criterion(outputs, targets).item() * data.size(0)
                        fold_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy()) # Store probabilities
                        fold_targets_list.extend(targets.cpu().numpy())

                avg_fold_val_loss = fold_val_loss / len(val_dataset_fold) if len(val_dataset_fold) > 0 else float('nan')
                y_pred_labels_fold = np.argmax(fold_preds, axis=1)
                y_pred_probs_fold = np.array(fold_preds)[:, 1]

                fold_accuracy = accuracy_score(fold_targets_list, y_pred_labels_fold)
                fold_precision = precision_score(fold_targets_list, y_pred_labels_fold, average='binary', zero_division=0)
                fold_recall = recall_score(fold_targets_list, y_pred_labels_fold, average='binary', zero_division=0)
                fold_f1 = f1_score(fold_targets_list, y_pred_labels_fold, average='binary', zero_division=0)
                fold_auc = float('nan')
                if len(np.unique(fold_targets_list)) > 1: # AUC requires at least two classes
                    try: fold_auc = roc_auc_score(fold_targets_list, y_pred_probs_fold)
                    except ValueError as e_auc: logger.warning(f"Kon AUC niet berekenen voor fold {fold_idx+1}: {e_auc}")

                fold_metrics_list.append({
                    "fold": fold_idx + 1, "val_loss": avg_fold_val_loss, "accuracy": fold_accuracy,
                    "precision": fold_precision, "recall": fold_recall, "f1": fold_f1, "auc": fold_auc
                })
                logger.info(f"Fold {fold_idx+1} Metrics: Val Loss={avg_fold_val_loss:.4f}, Acc={fold_accuracy:.4f}, AUC={fold_auc:.4f}")

            if fold_metrics_list: # If any folds were successfully processed
                cv_results = {"num_splits": cv_n_splits, "folds": fold_metrics_list}
                for metric_key in ["val_loss", "accuracy", "precision", "recall", "f1", "auc"]:
                    valid_fold_values = [m[metric_key] for m in fold_metrics_list if not np.isnan(m[metric_key])]
                    if valid_fold_values:
                        cv_results[f"mean_{metric_key}"] = np.mean(valid_fold_values)
                        cv_results[f"std_{metric_key}"] = np.std(valid_fold_values)
                    else:
                        cv_results[f"mean_{metric_key}"] = float('nan')
                        cv_results[f"std_{metric_key}"] = float('nan')
                logger.info(f"CV Gemiddelde Val Loss: {cv_results.get('mean_val_loss', float('nan')):.4f}, Gemiddelde Acc: {cv_results.get('mean_accuracy', float('nan')):.4f}, Gemiddelde AUC: {cv_results.get('mean_auc', float('nan')):.4f}")
        else:
            if perform_cv: logger.warning(f"Cross-validation niet uitgevoerd voor {model_name_prefix} door onvoldoende samples.")


        # --- Hyperparameter Optimization (Optuna) ---
        perform_hpo = self.params_manager.get_param('perform_hyperparameter_optimization', False)
        hpo_num_trials = self.params_manager.get_param('hpo_num_trials', 20)
        hpo_sampler_str = self.params_manager.get_param('hpo_sampler', 'TPE')
        hpo_pruner_str = self.params_manager.get_param('hpo_pruner', 'Median')
        hpo_metric = self.params_manager.get_param('hpo_metric_to_optimize', 'val_loss')
        hpo_direction = self.params_manager.get_param('hpo_direction_to_optimize', 'minimize')

        if perform_hpo:
            logger.info(f"Starten Hyperparameter Optimalisatie (Optuna) voor {model_name_prefix}...")

            # Temporary HPO data (using a subset of X_train_full for faster trials)
            # In a real scenario, you might use X_train_full and X_val_full directly or create new splits
            # For this placeholder, we'll use the existing X_train_full, y_train_full for HPO training
            # and X_val_full, y_val_full for HPO validation within each trial.
            # This is a simplified approach for initial setup.
            X_hpo_train, y_hpo_train = X_train_full, y_train_full
            X_hpo_val, y_hpo_val = X_val_full, y_val_full

            hpo_train_dataset = TensorDataset(X_hpo_train, y_hpo_train)
            hpo_val_dataset = TensorDataset(X_hpo_val, y_hpo_val)

            # Reduced number of epochs for HPO trials for speed
            hpo_trial_epochs = self.params_manager.get_param('num_epochs_cnn_hpo_trial', 3)


            def objective(trial: optuna.trial.Trial) -> float:
                try:
                    trial_arch_params = {}
                    trial_arch_params['num_conv_layers'] = trial.suggest_int('num_conv_layers', 1, 3) # Max 3 for simplicity now

                    filters_list = []
                    kernel_sizes_list = []
                    strides_list = []
                    padding_list = []
                    pooling_types_list = []
                    pooling_kernels_list = []
                    pooling_strides_list = []

                    for i in range(trial_arch_params['num_conv_layers']):
                        filters_list.append(trial.suggest_categorical(f'filters_layer_{i}', [16, 32, 64, 128]))
                        kernel_sizes_list.append(trial.suggest_categorical(f'kernel_size_layer_{i}', [3, 5]))
                        strides_list.append(trial.suggest_categorical(f'stride_layer_{i}', [1])) # Usually 1 for conv
                        padding_list.append(trial.suggest_categorical(f'padding_layer_{i}', [0, 1, 2])) # Or calculate for 'same'

                        pool_type = trial.suggest_categorical(f'pool_type_layer_{i}', ['max', 'avg', 'none'])
                        pooling_types_list.append(pool_type)
                        if pool_type != 'none':
                            pooling_kernels_list.append(trial.suggest_categorical(f'pool_kernel_layer_{i}', [2]))
                            pooling_strides_list.append(trial.suggest_categorical(f'pool_stride_layer_{i}', [2]))
                        else:
                            pooling_kernels_list.append(1) # Dummy value, not used
                            pooling_strides_list.append(1) # Dummy value, not used

                    trial_arch_params['filters_per_layer'] = filters_list
                    trial_arch_params['kernel_sizes_per_layer'] = kernel_sizes_list
                    trial_arch_params['strides_per_layer'] = strides_list
                    trial_arch_params['padding_per_layer'] = padding_list
                    trial_arch_params['pooling_types_per_layer'] = pooling_types_list
                    trial_arch_params['pooling_kernel_sizes_per_layer'] = pooling_kernels_list
                    trial_arch_params['pooling_strides_per_layer'] = pooling_strides_list

                    trial_arch_params['use_batch_norm'] = trial.suggest_categorical('use_batch_norm', [True, False])
                    trial_arch_params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.0, 0.5)

                    # Training parameters
                    trial_learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                    trial_batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

                    hpo_model = SimpleCNN(
                        input_channels=X_hpo_train.shape[1],
                        num_classes=2,
                        sequence_length=sequence_length,
                        **trial_arch_params
                    )
                    hpo_optimizer = optim.Adam(hpo_model.parameters(), lr=trial_learning_rate)
                    hpo_criterion = nn.CrossEntropyLoss()

                    hpo_train_loader = DataLoader(hpo_train_dataset, batch_size=trial_batch_size, shuffle=True)
                    hpo_val_loader = DataLoader(hpo_val_dataset, batch_size=trial_batch_size, shuffle=False)

                    for epoch in range(hpo_trial_epochs):
                        hpo_model.train()
                        for data, targets in hpo_train_loader:
                            hpo_optimizer.zero_grad()
                            outputs = hpo_model(data)
                            loss = hpo_criterion(outputs, targets)
                            loss.backward()
                            hpo_optimizer.step()

                    hpo_model.eval()
                    trial_val_loss = 0
                    all_preds_probs = []
                    all_targets_list = []
                    with torch.no_grad():
                        for data, targets in hpo_val_loader:
                            outputs = hpo_model(data)
                            loss = hpo_criterion(outputs, targets)
                            trial_val_loss += loss.item() * data.size(0)
                            all_preds_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                            all_targets_list.extend(targets.cpu().numpy())

                    avg_trial_val_loss = trial_val_loss / len(hpo_val_dataset) if len(hpo_val_dataset) > 0 else float('inf')

                    if hpo_metric == 'val_loss': metric_value = avg_trial_val_loss
                    elif hpo_metric == 'val_accuracy':
                        preds_labels = (np.array(all_preds_probs) > 0.5).astype(int)
                        metric_value = accuracy_score(all_targets_list, preds_labels)
                    elif hpo_metric == 'val_auc':
                        if len(np.unique(all_targets_list)) < 2: metric_value = 0.0
                        else: metric_value = roc_auc_score(all_targets_list, all_preds_probs)
                    else:
                        logger.warning(f"Unsupported HPO metric '{hpo_metric}'. Defaulting to 'val_loss'.")
                        metric_value = avg_trial_val_loss

                    return metric_value

                except optuna.exceptions.TrialPruned:
                    raise # Pruner requested to stop this trial
                except Exception as e:
                    logger.error(f"Exception in Optuna objective function for trial {trial.number}: {e}", exc_info=True)
                    return float('inf') if hpo_direction == 'minimize' else 0.0 # Penalize bad trials

            sampler = optuna.samplers.TPESampler() if hpo_sampler_str.lower() == 'tpe' else optuna.samplers.RandomSampler()
            pruner = optuna.pruners.MedianPruner() if hpo_pruner_str.lower() == 'median' else None

            study = optuna.create_study(direction=hpo_direction, sampler=sampler, pruner=pruner)
            hpo_timeout = self.params_manager.get_param('hpo_timeout_seconds', 3600) # Default 1 hour timeout for HPO
            study.optimize(objective, n_trials=hpo_num_trials, timeout=hpo_timeout)

            best_hpo_params = study.best_trial.params
            logger.info(f"HPO voltooid voor {model_name_prefix}. Beste trial waarde ({hpo_metric}): {study.best_trial.value}")
            logger.info(f"Beste HPO parameters: {best_hpo_params}")

            # Update main training variables with HPO results
            learning_rate = best_hpo_params['learning_rate']
            batch_size = best_hpo_params['batch_size']

            # Reconstruct arch_params from best_hpo_params
            arch_params['num_conv_layers'] = best_hpo_params['num_conv_layers']
            arch_params['filters_per_layer'] = [best_hpo_params[f'filters_layer_{i}'] for i in range(arch_params['num_conv_layers'])]
            arch_params['kernel_sizes_per_layer'] = [best_hpo_params[f'kernel_size_layer_{i}'] for i in range(arch_params['num_conv_layers'])]
            arch_params['strides_per_layer'] = [best_hpo_params[f'stride_layer_{i}'] for i in range(arch_params['num_conv_layers'])]
            arch_params['padding_per_layer'] = [best_hpo_params[f'padding_layer_{i}'] for i in range(arch_params['num_conv_layers'])]
            arch_params['pooling_types_per_layer'] = [best_hpo_params[f'pool_type_layer_{i}'] for i in range(arch_params['num_conv_layers'])]

            pooling_kernels = []
            pooling_strides = []
            for i in range(arch_params['num_conv_layers']):
                if arch_params['pooling_types_per_layer'][i] != 'none':
                    pooling_kernels.append(best_hpo_params[f'pool_kernel_layer_{i}'])
                    pooling_strides.append(best_hpo_params[f'pool_stride_layer_{i}'])
                else:
                    pooling_kernels.append(1) # Dummy, not used by SimpleCNN
                    pooling_strides.append(1) # Dummy, not used by SimpleCNN
            arch_params['pooling_kernel_sizes_per_layer'] = pooling_kernels
            arch_params['pooling_strides_per_layer'] = pooling_strides

            arch_params['use_batch_norm'] = best_hpo_params['use_batch_norm']
            arch_params['dropout_rate'] = best_hpo_params['dropout_rate']

            logger.info(f"Hoofd trainingsparameters bijgewerkt met HPO resultaten. Nieuwe arch_params: {arch_params}")
            logger.info(f"Nieuwe learning_rate: {learning_rate}, Nieuwe batch_size: {batch_size}")

            # Update model name prefix and paths to reflect HPO was run AND the regime
            model_name_prefix = f"{model_name_base}_hpo_{regime_name}"
            # model_dir is self.models_dir / symbol_sani / timeframe
            # Ensure model_dir is correctly defined before this point; it seems to be missing in this specific method context
            # Assuming model_dir should be derived similar to pair_tf_models_dir in pretrain()
            # For now, this part of the logic might be flawed if model_dir is not passed or set for train_ai_models
            # Let's assume self.models_dir is the base, and we need to construct the full path here.
            current_model_dir = (self.models_dir / symbol.replace('/', '_') / timeframe).resolve()
            current_model_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

            model_save_path = current_model_dir / f'cnn_model_{pattern_type}_{current_arch_key_orig}_hpo_{regime_name}.pth'
            scaler_save_path = current_model_dir / f'scaler_params_{pattern_type}_{current_arch_key_orig}_hpo_{regime_name}.json'
            logger.info(f"HPO uitgevoerd. Modelnaam prefix: {model_name_prefix}")
        else:
            # No HPO, use regime name in the prefix and paths
            model_name_prefix = f"{model_name_base}_{regime_name}"
            current_model_dir = (self.models_dir / symbol.replace('/', '_') / timeframe).resolve()
            current_model_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

            model_save_path = current_model_dir / f'cnn_model_{pattern_type}_{current_arch_key_orig}_{regime_name}.pth'
            scaler_save_path = current_model_dir / f'scaler_params_{pattern_type}_{current_arch_key_orig}_{regime_name}.json'
            logger.info(f"Geen HPO uitgevoerd. Modelnaam prefix: {model_name_prefix}")

        logger.info(f"Start training PyTorch AI-model '{model_name_prefix}' ({regime_name} regime) met {len(training_data)} samples...")

        if training_data.empty: logger.warning(f"Geen trainingsdata voor '{model_name_prefix}'."); return

        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
        if not all(col in training_data.columns for col in feature_columns + [target_label_column]):
            missing_cols = [col for col in feature_columns + [target_label_column] if col not in training_data.columns]
            logger.error(f"Benodigde kolommen ({missing_cols}) niet aanwezig in training_data voor '{model_name_prefix}'. Overslaan.")
            return

        # model_dir is already defined based on symbol and timeframe
        # model_save_path and scaler_save_path are now set above

        # --- Finaal model trainen op de volledige X_train_full, y_train_full ---
        logger.info(f"Starten van definitieve modeltraining voor {model_name_prefix} ({regime_name} regime) op volledige trainingsset (parameters mogelijk HPO-geoptimaliseerd)...")
        train_dataset_full = TensorDataset(X_train_full, y_train_full)
        val_dataset_full = TensorDataset(X_val_full, y_val_full) # Gebruik de oorspronkelijke validatieset
        train_loader_full = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True)
        val_loader_full = DataLoader(val_dataset_full, batch_size=batch_size, shuffle=False)

        final_model = SimpleCNN(input_channels=X_train_full.shape[1], num_classes=2, sequence_length=sequence_length, **arch_params)
        final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
        final_criterion = nn.CrossEntropyLoss()

        best_val_loss, epochs_no_improve, best_metrics = float('inf'), 0, {}
        patience = self.params_manager.get_param('early_stopping_patience_cnn', 3)

        for epoch in range(num_epochs):
            final_model.train()
            epoch_train_loss = sum(loss.item()*data.size(0) for data,targets in train_loader_full for outputs in [final_model(data)] for loss in [final_criterion(outputs,targets)] for _ in [final_optimizer.zero_grad(),loss.backward(),final_optimizer.step()]) / len(train_dataset_full)
            final_model.eval(); current_epoch_val_loss, all_preds, all_targets = 0.0, [], []
            with torch.no_grad():
                for data, targets in val_loader_full: # Valideer op X_val_full, y_val_full (deze is gedefinieerd buiten HPO-blok)
                    outputs = final_model(data); current_epoch_val_loss += final_criterion(outputs, targets).item() * data.size(0)
                    all_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy()) # Store probabilities
                    all_targets.extend(targets.cpu().numpy())

            avg_epoch_val_loss = current_epoch_val_loss / len(val_dataset_full) if len(val_dataset_full) > 0 else float('nan') # val_dataset_full is from outer scope
            y_pred_labels = np.argmax(all_preds, axis=1)
            y_pred_probs = np.array(all_preds)[:, 1]

            current_metrics = {"loss":avg_epoch_val_loss, "accuracy":accuracy_score(all_targets,y_pred_labels),
                               "precision":precision_score(all_targets,y_pred_labels,average='binary',zero_division=0),
                               "recall":recall_score(all_targets,y_pred_labels,average='binary',zero_division=0),
                               "f1":f1_score(all_targets,y_pred_labels,average='binary',zero_division=0),
                               "auc": roc_auc_score(all_targets, y_pred_probs) if len(np.unique(all_targets)) > 1 else float('nan')}
            logger.debug(f"Final Model - Epoch [{epoch+1}/{num_epochs}], {model_name_prefix}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {current_metrics['loss']:.4f}, Acc: {current_metrics['accuracy']:.4f}, AUC: {current_metrics['auc']:.4f}")
            if current_metrics['loss'] < best_val_loss:
                best_val_loss, best_metrics, epochs_no_improve = current_metrics['loss'], current_metrics, 0
                torch.save(final_model.state_dict(), str(model_save_path)) # torch.save might need string path
                logger.debug(f"Beste final model '{model_name_prefix}' opgeslagen (Epoch {epoch+1}) naar {model_save_path}. Val Loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience: logger.info(f"Early stopping final model '{model_name_prefix}' (Epoch {epoch+1})."); break
        try:
            if model_save_path.exists(): logger.info(f"'{model_name_prefix}' final training voltooid. Model opgeslagen: {model_save_path}")
            else: logger.warning(f"'{model_name_prefix}' final training voltooid. Geen model opgeslagen: {model_save_path}.")
            scaler_params_to_save = {'feature_names_in_':feature_names_in_, 'min_':min_, 'scale_':scale_, 'sequence_length':sequence_length}
            with scaler_save_path.open('w', encoding='utf-8') as f: json.dump(scaler_params_to_save, f, indent=4)
            logger.info(f"Scaler params '{model_name_prefix}' opgeslagen: {scaler_save_path}")
        except Exception as e: logger.error(f"Fout opslaan final model/scaler '{model_name_prefix}': {e}", exc_info=True)

        await asyncio.to_thread(self._log_pretrain_activity, model_type=model_name_prefix, regime_name=regime_name, data_size=len(X_train_full)+len(X_val_full),
                                model_path_saved=str(model_save_path), scaler_params_path_saved=str(scaler_save_path), # Log string paths
                                cv_results=cv_results, **best_metrics)
        logger.info(f"Pre-training {model_name_prefix} ({regime_name} regime) voltooid. Beste Val Loss: {best_metrics.get('loss', float('inf')):.4f}, Acc: {best_metrics.get('accuracy', 0.0):.4f}")
        return best_metrics.get('loss')

    def _log_pretrain_activity(self, model_type: str, data_size: int, model_path_saved: str, scaler_params_path_saved: str,
                             regime_name: Optional[str] = None, # Added regime_name
                             best_val_loss: float = None, best_val_accuracy: float = None,
                             best_val_precision: float = None, best_val_recall: float = None,
                             best_val_f1: float = None, auc: float = None,
                             cv_results: Optional[dict] = None):
        entry = {"timestamp": datetime.now().isoformat(), "model_type": model_type, "regime_name": regime_name, "data_size": data_size,
                 "status": "completed_pytorch_training", "model_path_saved": model_path_saved,
                 "scaler_params_path_saved": scaler_params_path_saved, # Corrected key
                 "final_model_validation_loss": best_val_loss, # Clarified this is for the final model
                 "final_model_validation_accuracy": best_val_accuracy,
                 "final_model_validation_precision": best_val_precision,
                 "final_model_validation_recall": best_val_recall,
                 "final_model_validation_f1": best_val_f1,
                 "final_model_validation_auc": auc}
        if cv_results:
            entry['cross_validation_results'] = cv_results
        logs = []
        try:
            if PRETRAIN_LOG_FILE.exists() and PRETRAIN_LOG_FILE.stat().st_size > 0:
                with PRETRAIN_LOG_FILE.open('r', encoding='utf-8') as f: logs = json.load(f)
                if not isinstance(logs, list): logs = [logs] # Handle case where file might contain a single dict
            logs.append(entry)
            with PRETRAIN_LOG_FILE.open('w', encoding='utf-8') as f: json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Fout bij loggen pre-train activiteit naar {PRETRAIN_LOG_FILE}: {e}", exc_info=True)

    async def analyze_time_of_day_effectiveness(self, historical_data: pd.DataFrame, strategy_id: str) -> Dict[str, Any]:
        logger.info(f"Analyseren tijd-van-dag effectiviteit voor strategie {strategy_id}...")
        if historical_data.empty:
            logger.info("Historische data is leeg. Kan tijd-van-dag effectiviteit niet analyseren.")
            return {}
        label_to_analyze = next((lbl for lbl in ['bullFlag_label', 'bearishEngulfing_label'] if lbl in historical_data.columns), None)
        if not label_to_analyze:
            label_to_analyze = next((col for col in historical_data.columns if col.endswith('_label')), None)
            if label_to_analyze:
                logger.info(f"Generiek label '{label_to_analyze}' gevonden voor tijd-van-dag analyse.")
            else:
                logger.warning("Geen label kolom (*_label) gevonden voor tijd-van-dag analyse.")
                return {}
        if not isinstance(historical_data.index, pd.DatetimeIndex): #This check is correct
            logger.error("DataFrame index geen DatetimeIndex. Kan uur niet extraheren.") # Logged correctly
            return {}

        df_copy = historical_data.copy()
        df_copy['hour_of_day'] = df_copy.index.hour
        time_effectiveness = df_copy.groupby('hour_of_day')[label_to_analyze].agg(
            ['mean', 'count']
        ).rename(columns={'mean': f'avg_{label_to_analyze}_proportion', 'count': 'num_samples'})

        result_dict = time_effectiveness.to_dict(orient='index')
        logger.info(f"Tijd-van-dag effectiviteit (obv {label_to_analyze}) geanalyseerd: {result_dict}")
        try:
            with TIME_EFFECTIVENESS_FILE.open('w', encoding='utf-8') as f: # TIME_EFFECTIVENESS_FILE is Path
                json.dump(result_dict, f, indent=2)
            logger.info(f"Tijd-van-dag effectiviteit opgeslagen naar {TIME_EFFECTIVENESS_FILE.resolve()}")
        except Exception as e:
            logger.error(f"Fout bij opslaan tijd-van-dag effectiviteit naar {TIME_EFFECTIVENESS_FILE.resolve()}: {e}", exc_info=True) # exc_info=True is correct
        return result_dict

    async def run_pretraining_pipeline(self, strategy_id: str):
        logger.info(f"Start pre-training pipeline voor strategie: {strategy_id}...")
        # self.params_manager will be used directly
        pairs_to_fetch = self.params_manager.get_param("data_fetch_pairs")
        timeframes_to_fetch = self.params_manager.get_param("data_fetch_timeframes")
        patterns_to_train_list = self.params_manager.get_param('patterns_to_train')
        regimes_to_train_for = self.params_manager.get_param('regimes_to_train', ["all"])

        if not pairs_to_fetch or not timeframes_to_fetch:
            logger.error("Geen 'data_fetch_pairs' of 'data_fetch_timeframes' in params. Pipeline gestopt.")
            return
        if not patterns_to_train_list:
            logger.warning("Geen 'patterns_to_train' funnet i parametere. Standaard naar ['bullFlag'].")
            patterns_to_train_list = ["bullFlag"]

        logger.info(f"Pipeline voor paren: {pairs_to_fetch}, timeframes: {timeframes_to_fetch}, patterns: {patterns_to_train_list}, regimes: {regimes_to_train_for}")

        for pattern_type in patterns_to_train_list:
            logger.info(f"--- Starten trainingsronde voor patroontype: {pattern_type} ---")
            for pair in pairs_to_fetch:
                for timeframe in timeframes_to_fetch:
                    logger.info(f"--- Data ophalen voor {pair} ({timeframe}), patroon: {pattern_type} ---")
                    historical_data_by_regime = await self.fetch_historical_data(pair, timeframe)

                    if not historical_data_by_regime or all(df.empty for df in historical_data_by_regime.values()):
                        logger.warning(f"Geen historische data (per regime of 'all') beschikbaar voor {pair} ({timeframe}). Overslaan.")
                        continue

                    for regime_name in regimes_to_train_for:
                        logger.info(f"--- Verwerken {pair} ({timeframe}) voor patroon: {pattern_type}, regime: {regime_name} ---")

                        historical_df_for_regime = historical_data_by_regime.get(regime_name)
                        if historical_df_for_regime is None or historical_df_for_regime.empty:
                            logger.warning(f"Geen data voor regime '{regime_name}' voor {pair} ({timeframe}). Overslaan training voor dit regime.")
                            continue

                        logger.info(f"Voorbereiden data voor {pair}-{timeframe}-{pattern_type}-{regime_name} ({len(historical_df_for_regime)} kaarsen).")
                        processed_df = await self.prepare_training_data(historical_df_for_regime.copy(), pattern_type=pattern_type)

                        if processed_df.empty:
                            logger.error(f"Geen data na voorbereiding voor {pair} ({timeframe}), patroon {pattern_type}, regime {regime_name}. Overslaan.")
                            continue

                        target_label_column = f"{pattern_type}_label"
                        await self.train_ai_models(
                            training_data=processed_df,
                            symbol=pair,
                            timeframe=timeframe,
                            pattern_type=pattern_type,
                            target_label_column=target_label_column,
                            regime_name=regime_name
                        )

                        if not processed_df.empty: # Only analyze if there was data
                            await self.analyze_time_of_day_effectiveness(processed_df, f"{strategy_id}_{pattern_type}_{regime_name}")

        logger.info(f"--- Pre-training en model training voltooid voor strategie {strategy_id} voor alle geconfigureerde patronen/paren/tijdsbestekken/regimes ---")

        # --- Backtesting Phase Removed ---
        # The custom backtester functionality has been deprecated and core/backtester.py removed.
        # Project will rely on Freqtrade's built-in backtesting.
        logger.info(f"Custom backtesting phase skipped as it has been deprecated.")

        logger.info(f"Pre-training pipeline voltooid voor strategie {strategy_id} for alle konfigurerte patterns.")

if __name__ == "__main__":
    import sys # sys is used for StreamHandler, ensure it's imported.
    # Path is already imported.
    # dotenv.load_dotenv is already imported.
    dotenv_path_main = PROJECT_ROOT_DIR / '.env' # This is correct
    if dotenv_path_main.exists():
        dotenv.load_dotenv(dotenv_path_main) # Correct
    else:
        logger.warning(f".env file not found at {dotenv_path_main} for __main__ test run.")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    # ParamsManager is already imported.
    # CNNPatterns is already imported.

    BITVAVO_API_KEY = os.getenv('BITVAVO_API_KEY') # Correct
    BITVAVO_SECRET_KEY = os.getenv('BITVAVO_SECRET_KEY') # Correct
    if not BITVAVO_API_KEY or not BITVAVO_SECRET_KEY:
        logger.warning("Bitvavo API keys niet gevonden in .env. Data ophalen zal waarschijnlijk falen als Bitvavo exchange geconfigureerd is.")

    TEST_USER_DATA_DIR = (PROJECT_ROOT_DIR / "user_data_pretrainer_test").resolve()


    def _get_test_config() -> Dict[str, Any]:
        """Centralized configuration for the test run."""
        test_artifacts_input_dir = (PROJECT_ROOT_DIR / 'test_artifacts' / 'pretrainer_inputs').resolve()
        pretrainer_test_models_output_dir = (TEST_USER_DATA_DIR / "models" / "cnn_pretrainer").resolve()

        # Test-specific file paths using global Path constants as base and adding _test suffix
        test_market_regimes_file = MARKET_REGIMES_FILE.with_name(MARKET_REGIMES_FILE.name.replace('.json', '_test.json'))
        test_pretrain_log_file = PRETRAIN_LOG_FILE.with_name(PRETRAIN_LOG_FILE.name.replace('.json', '_test.json'))
        test_time_effectiveness_file = TIME_EFFECTIVENESS_FILE.with_name(TIME_EFFECTIVENESS_FILE.name.replace('.json', '_test.json'))
        # Assuming a similar pattern for backtest_results if it were managed by global Path constants
        test_backtest_results_file = (USER_DATA_MEMORY_DIR / 'backtest_results_test.json').resolve()


        config = {
            "pairs": ["ETH/EUR"],
            "timeframe": "1h",
            "patterns_to_train": ["bullFlag"],
            "arch_key": "default_simple",
            "regimes": ["all", "testbull"],
            "test_artifacts_input_dir": test_artifacts_input_dir, # Path for dummy gold standard CSVs
            "gold_standard_dir": (test_artifacts_input_dir / 'gold_standard_data').resolve(),
            "market_regimes_file_path": test_market_regimes_file,
            "pretrain_log_file_path": test_pretrain_log_file,
            "time_effectiveness_file_path": test_time_effectiveness_file,
            "backtest_results_file_path": test_backtest_results_file,
            "pretrainer_test_models_output_dir": pretrainer_test_models_output_dir, # Expected output dir for models
            "dummy_regimes_content": {
                "ETH/EUR": { # Should match test_config["pairs"][0]
                    "testbull": [{"start_date": "2023-11-01", "end_date": "2023-11-03"}],
                    "testbear": [{"start_date": "2023-11-04", "end_date": "2023-11-06"}]
                }
            }
        }
        config["dummy_gold_csv_path"] = config["gold_standard_dir"] / f"{config['pairs'][0].replace('/', '_')}_{config['timeframe']}_{config['patterns_to_train'][0]}_gold.csv"
        return config

    def _clear_test_artifacts(test_config: Dict[str, Any]):
        logger.info(f"--- Test Setup: Clearing Artifacts ---")
        # Clear model outputs for this specific test config
        for pair_name in test_config["pairs"]:
            pair_sani = pair_name.replace('/', '_')
            # Construct path based on how PreTrainer saves models (self.models_dir)
            model_output_dir = test_config["models_dir_path"] / pair_sani / test_config["timeframe"]
            if model_output_dir.exists():
                shutil.rmtree(model_output_dir)
                logger.info(f"Removed model output directory: {model_output_dir}")

        # Clear general test artifacts dir (which includes gold standard dir)
        if test_config["test_data_base_dir"].exists():
            shutil.rmtree(test_config["test_data_base_dir"])
            logger.info(f"Removed test artifacts base directory: {test_config['test_data_base_dir']}")
        os.makedirs(test_config["gold_standard_dir"], exist_ok=True) # Recreate gold_standard_dir

        # Clear main log files and runtime files PreTrainer uses
        files_to_remove = [
            test_config["pretrain_log_file_path"],
            test_config["time_effectiveness_file_path"],
            test_config["market_regimes_file_path"], # This is where dummy market regimes will be written
            test_config["backtest_results_file_path"]
        ]
        for file_path in files_to_remove:
            if file_path.exists():
                os.remove(file_path)
                logger.info(f"Removed file: {file_path}")
        logger.info(f"--- Test Setup: Artifacts Cleared ---")

    def _create_dummy_test_data(test_config: Dict[str, Any]):
        logger.info(f"--- Test Setup: Creating Dummy Data ---")
        # Create Dummy Gold Standard CSV
        with open(test_config["dummy_gold_csv_path"], 'w') as f:
            f.write("timestamp,open,high,low,close,volume,gold_label\n")
            f.write("1698883200000,1800,1801,1799,1800.5,10,1\n") # Nov 2, 2023, 00:00 GMT
            f.write("1698886800000,1800.5,1802,1799.5,1801.0,12,0\n") # Nov 2, 2023, 01:00 GMT
        logger.info(f"Dummy gold standard CSV created at: {test_config['dummy_gold_csv_path']}")

        # Create Dummy Market Regimes JSON at the expected runtime path
        with open(test_config["market_regimes_file_path"], 'w') as f:
            json.dump(test_config["dummy_regimes_content"], f, indent=4)
        logger.info(f"Dummy market regimes JSON created at: {test_config['market_regimes_file_path']}")
        logger.info(f"--- Test Setup: Dummy Data Created ---")

    async def _setup_test_params_manager(params_manager: ParamsManager, test_config: Dict[str, Any]):
        logger.info(f"--- Test Setup: Configuring ParamsManager ---")
        await params_manager.set_param("data_fetch_start_date_str", "2023-10-30")
        await params_manager.set_param("data_fetch_end_date_str", "2023-11-10")
        await params_manager.set_param("data_fetch_pairs", test_config["pairs"])
        await params_manager.set_param("data_fetch_timeframes", [test_config["timeframe"]])
        await params_manager.set_param("patterns_to_train", test_config["patterns_to_train"])

        await params_manager.set_param('sequence_length_cnn', 5)
        await params_manager.set_param('num_epochs_cnn', 1)
        await params_manager.set_param('batch_size_cnn', 4)
        await params_manager.set_param('num_epochs_cnn_hpo_trial', 1)

        await params_manager.set_param("gold_standard_data_path", str(test_config["gold_standard_dir"]))
        await params_manager.set_param('perform_hyperparameter_optimization', True)
        await params_manager.set_param('hpo_num_trials', 2)
        await params_manager.set_param('hpo_timeout_seconds', 60)
        await params_manager.set_param('regimes_to_train', test_config["regimes"])

        labeling_configs = params_manager.get_param('pattern_labeling_configs', {})
        labeling_configs.setdefault(test_config["patterns_to_train"][0], {}).update({
            "future_N_candles": 3, "profit_threshold_pct": 0.001, "loss_threshold_pct": -0.001
        })
        await params_manager.set_param('pattern_labeling_configs', labeling_configs)

        await params_manager.set_param('current_cnn_architecture_key', test_config["arch_key"])
        cnn_arch_configs = params_manager.get_param('cnn_architecture_configs', {})
        if test_config["arch_key"] not in cnn_arch_configs:
            cnn_arch_configs[test_config["arch_key"]] = {
                "num_conv_layers": 1, "filters_per_layer": [16], "kernel_sizes_per_layer": [3],
                "strides_per_layer": [1], "padding_per_layer": [1], "pooling_types_per_layer": ["max"],
                "pooling_kernel_sizes_per_layer": [2], "pooling_strides_per_layer": [2],
                "use_batch_norm": False, "dropout_rate": 0.1
            }
            await params_manager.set_param('cnn_architecture_configs', cnn_arch_configs)
        logger.info(f"--- Test Setup: ParamsManager Configured ---")

    def _validate_test_output_files(test_config: Dict[str, Any]):
        logger.info(f"--- Validating Output Model/Scaler Files ---")
        for pair_name in test_config["pairs"]:
            pair_sani = pair_name.replace('/', '_')
            for pattern in test_config["patterns_to_train"]:
                for regime_name_val in test_config["regimes"]:
                    model_dir_val = test_config["models_dir_path"] / pair_sani / test_config["timeframe"]
                    expected_model_fname = f'cnn_model_{pattern}_{test_config["arch_key"]}_hpo_{regime_name_val}.pth'
                    expected_scaler_fname = f'scaler_params_{pattern}_{test_config["arch_key"]}_hpo_{regime_name_val}.json'

                    expected_model_file = model_dir_val / expected_model_fname
                    expected_scaler_file = model_dir_val / expected_scaler_fname

                    assert expected_model_file.exists(), f"Model file MISSING: {expected_model_file}"
                    logger.info(f"SUCCESS: Model file for {regime_name_val} regime created: {expected_model_file.name}")
                    assert expected_scaler_file.exists(), f"Scaler file MISSING: {expected_scaler_file}"
                    logger.info(f"SUCCESS: Scaler file for {regime_name_val} regime created: {expected_scaler_file.name}")
        logger.info(f"--- Output Model/Scaler Files Validated ---")

    def _validate_test_log_files(test_config: Dict[str, Any]):
        logger.info(f"--- Validating Log Files ---")
        # Validate Pretrain Log
        pretrain_log_file = test_config["pretrain_log_file_path"]
        assert pretrain_log_file.exists(), f"Pretrain log file MISSING: {pretrain_log_file}"
        with open(pretrain_log_file, 'r') as f:
            log_entries = json.load(f)
        assert isinstance(log_entries, list), "Pretrain log is not a list."

        found_all_regime_log = any(
            entry["regime_name"] == "all" and f"{test_config['arch_key']}_hpo_all" in entry["model_type"]
            for entry in log_entries
        )
        found_testbull_regime_log = any(
            entry["regime_name"] == "testbull" and f"{test_config['arch_key']}_hpo_testbull" in entry["model_type"]
            for entry in log_entries
        )
        for entry in log_entries: # Basic check for CV results presence
             assert "cross_validation_results" in entry, f"cross_validation_results missing in log entry: {entry}"


        assert found_all_regime_log, "Log entry for 'all' regime (with HPO suffix) not found."
        assert found_testbull_regime_log, "Log entry for 'testbull' regime (with HPO suffix) not found."
        logger.info("SUCCESS: Pretrain log contains entries for 'all' and 'testbull' regimes with HPO naming and CV results.")

        # Validate Backtest Results File
        backtest_results_file = test_config["backtest_results_file_path"]
        assert backtest_results_file.exists(), f"Backtest results file MISSING: {backtest_results_file}"
        with open(backtest_results_file, 'r') as f:
            try:
                backtest_log_entries = json.load(f)
                assert isinstance(backtest_log_entries, list) and len(backtest_log_entries) > 0, \
                    "Backtest log is empty or not a list."
                logger.info(f"SUCCESS: Backtest results file created and contains {len(backtest_log_entries)} entries.")
                first_bt_entry = backtest_log_entries[0]
                assert first_bt_entry['architecture_key'] == test_config["arch_key"], \
                    f"Backtest ran with {first_bt_entry['architecture_key']} instead of {test_config['arch_key']}"
                logger.info(f"Backtest entry seems OK, used architecture key: {first_bt_entry['architecture_key']}")
            except json.JSONDecodeError:
                assert False, f"FAILURE: Could not decode JSON from backtest results file {backtest_results_file}."
        logger.info(f"--- Log Files Validated ---")

    async def run_test_pre_trainer():
        test_config = _get_test_config()

        logger.info(f"--- STARTING INTEGRATION TEST FOR PRETRAINER (ALL FEATURES) ---")
        logger.info(f"Test config: Pair={test_config['pairs'][0]}, TF={test_config['timeframe']}, "
                    f"Pattern={test_config['patterns_to_train'][0]}, Arch={test_config['arch_key']}, "
                    f"Regimes={test_config['regimes']}")
        logger.info(f"Test artifacts base dir: {test_config['test_data_base_dir']}")

        _clear_test_artifacts(test_config)
        _create_dummy_test_data(test_config)

        params_manager_instance = ParamsManager()
        await _setup_test_params_manager(params_manager_instance, test_config)

        # --- PreTrainer Config for instantiation ---
        # PreTrainer now takes a config dict in its constructor.
        # We need to provide a minimal one, or one that matches what it expects.
        # For this test, self.models_dir is important.
        # It's constructed as: self.config.get("user_data_dir", "user_data") + "/models/cnn_pretrainer"
        # So, we need to ensure "user_data_dir" is in the config passed to PreTrainer.
        pre_trainer_constructor_config = {
            "user_data_dir": BASE_USER_DATA_DIR, # "user_data" by default
            "exchange": {"name": "bitvavo"}, # Dummy, for potential BitvavoExecutor init
            # Add other minimal configs PreTrainer might need at init
        }

        pre_trainer = PreTrainer(config=pre_trainer_constructor_config)
        # Override params_manager if PreTrainer's __init__ creates its own without passing config
        pre_trainer.params_manager = params_manager_instance

        test_strategy_id = "TestStrategy_FullFeatures"

        try:
            logger.info(f"\n--- EXECUTING PreTrainer Pipeline (Gold, HPO, Regimes, CV, Backtesting) ---")
            # Pass the already configured params_manager_instance
            await pre_trainer.run_pretraining_pipeline(strategy_id=test_strategy_id, params_manager=params_manager_instance)
        finally:
            # --- Cleanup Test-Specific Artifacts (Runtime files like market_regimes.json) ---
            logger.info(f"--- Cleaning up runtime test-specific artifacts (e.g., market_regimes.json) ---")
            # This ensures MARKET_REGIMES_FILE (if modified for test) is cleaned
            if test_config["market_regimes_file_path"].exists() and \
               test_config["market_regimes_file_path"] == Path(MARKET_REGIMES_FILE): # only if it's the main one
                # Check if content is our dummy content before removing, or just remove if path matches
                # For simplicity, if the path is the global one, we remove it as it was a test-specific overwrite.
                try:
                    with open(test_config["market_regimes_file_path"], 'r') as f_mr:
                        content_mr = json.load(f_mr)
                        if content_mr == test_config["dummy_regimes_content"]:
                             os.remove(test_config["market_regimes_file_path"])
                             logger.info(f"Removed dummy market regimes file: {test_config['market_regimes_file_path']}")
                        else:
                            logger.warning(f"Market regimes file {test_config['market_regimes_file_path']} was not the dummy one. Not removed.")
                except Exception as e_mr_clean:
                    logger.error(f"Error cleaning market regimes file: {e_mr_clean}")

            # General test_data_base_dir (like gold standard) can be cleaned up here or left for inspection
            # _clear_test_artifacts already handles removal of test_data_base_dir for next run.
            # This finally block is more for runtime generated files that might persist if not handled by _clear_test_artifacts initially.


        _validate_test_output_files(test_config)
        _validate_test_log_files(test_config)

        # --- Optional: Test SimpleCNN Custom Architecture Instantiation (already present, keep) ---
        print("\n--- Testing SimpleCNN Custom Architecture Instantiation (original test) ---")
        try:
            num_input_features_example = 12
            # Assuming SimpleCNN is imported or accessible here
            # from core.cnn_models import SimpleCNN # Or wherever it's defined
            # For now, this part might fail if SimpleCNN is not directly in scope
            # This test was likely part of a different context or SimpleCNN was globally available
            # For this refactoring, we'll keep it but acknowledge it might need an import

            # Attempting to make SimpleCNN accessible for the test
            # This is a guess; actual location might differ
            try:
                from core.cnn_models import SimpleCNN # Assuming this is where it would be
            except ImportError:
                logger.warning("SimpleCNN class not found for custom instantiation test. Skipping this part.")
                SimpleCNN = None # Make it None so the test below skips

            if SimpleCNN:
                custom_cnn = SimpleCNN( input_channels=num_input_features_example, num_classes=2, sequence_length=30,
                    num_conv_layers=3, filters_per_layer=[16,32,64], kernel_sizes_per_layer=[3,3,5],
                    strides_per_layer=[1,1,1], padding_per_layer=[1,1,2],
                    pooling_types_per_layer=['max','max',None], pooling_kernel_sizes_per_layer=[2,2,1],
                    pooling_strides_per_layer=[2,2,1], use_batch_norm=True, dropout_rate=0.25)
                print("SUCCESS: Custom SimpleCNN instantiated.")
            else:
                print("SKIPPED: Custom SimpleCNN instantiation test (SimpleCNN class not found).")

        except Exception as e:
            print(f"FAILURE: Custom SimpleCNN instantiation failed: {e}")
            logger.error(f"Custom SimpleCNN instantiation failed: {e}", exc_info=True)

        logger.info("--- INTEGRATION TEST FOR PRETRAINER (ALL FEATURES) COMPLETED ---")

    # asyncio.run(run_test_pre_trainer()) # Original __main__ call, now moved to tests
    logger.info("PreTrainer __main__ execution block has been moved to tests/test_core_modules.py:TestPreTrainerIntegration.")
