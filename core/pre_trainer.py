# core/pre_trainer.py
import logging
import os
import json
from typing import Dict, Any, List, Optional # Added Optional
from datetime import datetime, timedelta
from datetime import datetime as dt
import pandas as pd
import numpy as np
import asyncio
from core.bitvavo_executor import BitvavoExecutor
from core.backtester import Backtester # Added Backtester import
import talib.abstract as ta
import dotenv
import shutil

# Scikit-learn imports
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Module-level docstring
"""
Handles the pre-training pipeline for CNN models based on historical market data.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PyTorch Imports
import torch
import torch.nn as nn
import optuna
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
PRETRAIN_LOG_FILE = os.path.join(MEMORY_DIR, 'pre_train_log.json')
TIME_EFFECTIVENESS_FILE = os.path.join(MEMORY_DIR, 'time_effectiveness.json')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'models')
MARKET_REGIMES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'market_regimes.json')

os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'), exist_ok=True)

class SimpleCNN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 sequence_length: int,
                 num_conv_layers: int = 2,
                 filters_per_layer: list = [16, 32],
                 kernel_sizes_per_layer: list = [3, 3],
                 strides_per_layer: list = [1, 1],
                 padding_per_layer: list = [1, 1],
                 pooling_types_per_layer: list = ['max', 'max'],
                 pooling_kernel_sizes_per_layer: list = [2, 2],
                 pooling_strides_per_layer: list = [2, 2],
                 use_batch_norm: bool = False,
                 dropout_rate: float = 0.0):
        super(SimpleCNN, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        param_lists = {
            "filters_per_layer": filters_per_layer, "kernel_sizes_per_layer": kernel_sizes_per_layer,
            "strides_per_layer": strides_per_layer, "padding_per_layer": padding_per_layer,
            "pooling_types_per_layer": pooling_types_per_layer,
            "pooling_kernel_sizes_per_layer": pooling_kernel_sizes_per_layer,
            "pooling_strides_per_layer": pooling_strides_per_layer
        }
        for name, p_list in param_lists.items():
            if len(p_list) != num_conv_layers:
                raise ValueError(f"Length of '{name}' ({len(p_list)}) must match 'num_conv_layers' ({num_conv_layers}).")
        self.conv_blocks = nn.ModuleList()
        current_channels = input_channels
        for i in range(num_conv_layers):
            block_layers = []
            block_layers.append(nn.Conv1d(in_channels=current_channels, out_channels=filters_per_layer[i], kernel_size=kernel_sizes_per_layer[i], stride=strides_per_layer[i], padding=padding_per_layer[i]))
            if use_batch_norm: block_layers.append(nn.BatchNorm1d(filters_per_layer[i]))
            block_layers.append(nn.ReLU())
            if pooling_types_per_layer[i] and pooling_types_per_layer[i].lower() != 'none':
                pool_kernel, pool_stride = pooling_kernel_sizes_per_layer[i], pooling_strides_per_layer[i]
                if pooling_types_per_layer[i].lower() == 'max': block_layers.append(nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride))
                elif pooling_types_per_layer[i].lower() == 'avg': block_layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride))
            self.conv_blocks.append(nn.Sequential(*block_layers)); current_channels = filters_per_layer[i]
        with torch.no_grad():
            dummy_x = torch.randn(1, self.input_channels, self.sequence_length)
            for block in self.conv_blocks: dummy_x = block(dummy_x)
            fc_input_features = dummy_x.view(dummy_x.size(0), -1).shape[1]
        self.dropout_layer = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.fc = nn.Linear(fc_input_features, num_classes)
    def forward(self, x):
        for block in self.conv_blocks: x = block(x)
        x = x.view(x.size(0), -1)
        if self.dropout_layer: x = self.dropout_layer(x)
        x = self.fc(x); return x

class PreTrainer:
    def __init__(self, params_manager=None):
        # ... (constructor as before, including CNNPatterns init) ...
        if params_manager is None:
            from core.params_manager import ParamsManager
            self.params_manager = ParamsManager()
            logger.info("ParamsManager niet meegegeven, standaard initialisatie gebruikt in PreTrainer.")
        else:
            self.params_manager = params_manager
        try:
            self.bitvavo_executor = BitvavoExecutor()
            logger.info("BitvavoExecutor geïnitialiseerd in PreTrainer.")
        except ValueError as e:
            logger.error(f"Fout bij initialiseren BitvavoExecutor: {e}. API keys mogelijk niet ingesteld.")
            self.bitvavo_executor = None

        self.market_regimes = {}
        try:
            if os.path.exists(MARKET_REGIMES_FILE):
                with open(MARKET_REGIMES_FILE, 'r') as f:
                    self.market_regimes = json.load(f)
                logger.info(f"Marktregimes geladen uit {MARKET_REGIMES_FILE}")
            else:
                logger.warning(f"{MARKET_REGIMES_FILE} niet gevonden. Er worden geen specifieke marktregimes gebruikt.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Fout bij laden/parsen {MARKET_REGIMES_FILE}: {e}.")
            self.market_regimes = {}

        from core.cnn_patterns import CNNPatterns
        self.cnn_pattern_detector = CNNPatterns()
        logger.info("PreTrainer geïnitialiseerd met CNNPatterns detector.")

        # Initialize Backtester
        self.backtester = Backtester(
            params_manager=self.params_manager,
            cnn_pattern_detector=self.cnn_pattern_detector,
            bitvavo_executor=self.bitvavo_executor
        )
        logger.info("Backtester geïnitialiseerd in PreTrainer.")


    def _get_cache_dir(self, symbol: str, timeframe: str) -> str:
        # ... (as before) ...
        symbol_sanitized = symbol.replace('/', '_')
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw_historical_data', symbol_sanitized, timeframe)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _get_cache_filepath(self, symbol: str, timeframe: str, start_dt: dt, end_dt: dt) -> str:
        # ... (as before) ...
        cache_dir = self._get_cache_dir(symbol, timeframe)
        symbol_sanitized = symbol.replace('/', '_')
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
        return os.path.join(cache_dir, f"{symbol_sanitized}_{timeframe}_{start_timestamp}_{end_timestamp}.csv")

    def _read_from_cache(self, filepath: str) -> pd.DataFrame | None:
        # ... (as before) ...
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, dtype={'timestamp': 'int64', 'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})
                expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in expected_cols):
                    logger.warning(f"Cache file {filepath} mangler kolonner. Forventet: {expected_cols}, Fikk: {df.columns.tolist()}. Ignorerer cache.")
                    return None
                logger.info(f"Leste {len(df)} candles fra cache: {filepath}")
                return df
            except Exception as e:
                logger.error(f"Feil ved lesing av cache-fil {filepath}: {e}. Ignorerer cache.")
                return None
        return None

    def _write_to_cache(self, filepath: str, data_df: pd.DataFrame):
        # ... (as before) ...
        try:
            expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data_df.columns for col in expected_cols):
                logger.error(f"DataFrame for caching mangler kolonner. Forventet: {expected_cols}, Fikk: {data_df.columns.tolist()}. Kan ikke cache.")
                return
            df_to_write = data_df[expected_cols].copy()
            df_to_write['timestamp'] = df_to_write['timestamp'].astype('int64')
            df_to_write.to_csv(filepath, index=False)
            logger.info(f"Skrev {len(data_df)} candles til cache: {filepath}")
        except Exception as e:
            logger.error(f"Feil ved skriving til cache-fil {filepath}: {e}")

    def _load_gold_standard_data(self, symbol: str, timeframe: str, pattern_type: str, expected_columns: List[str]) -> pd.DataFrame | None:
        """
        Loads gold standard data for a given symbol, timeframe, and pattern type.
        Verifies that the loaded data contains all expected columns and sets the timestamp as index.
        """
        gold_standard_path = self.params_manager.get_param("gold_standard_data_path")
        if not gold_standard_path:
            logger.debug("Gold standard data path not set in parameters. Skipping gold standard loading.")
            return None

        symbol_sanitized = symbol.replace('/', '_')
        filename = f"{symbol_sanitized}_{timeframe}_{pattern_type}_gold.csv"
        full_path = os.path.join(gold_standard_path, filename)

        if not os.path.exists(full_path):
            logger.debug(f"Gold standard data file not found: {full_path}")
            return None

        try:
            df = pd.read_csv(full_path)
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
            logger.debug(f"Gold standard data file not found (should have been caught by os.path.exists, but good to handle): {full_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.warning(f"Gold standard data file is empty (pandas error): {full_path}")
            return None
        except ValueError as ve:
            logger.error(f"ValueError processing gold standard data file {full_path}: {ve}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading gold standard data from {full_path}: {e}")
            return None

    async def _fetch_ohlcv_for_period(self, symbol: str, timeframe: str, start_dt: dt, end_dt: dt) -> pd.DataFrame | None:
        # ... (as before, with caching) ...
        cache_filepath = self._get_cache_filepath(symbol, timeframe, start_dt, end_dt)
        cached_df = self._read_from_cache(cache_filepath)
        if cached_df is not None:
            cached_df['date'] = pd.to_datetime(cached_df['timestamp'], unit='ms')
            cached_df.set_index('date', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in cached_df.columns:
                     cached_df[col] = pd.to_numeric(cached_df[col], errors='coerce')
            if 'timestamp' in cached_df.columns:
                 cached_df.drop(columns=['timestamp'], inplace=True)
            return cached_df
        if not self.bitvavo_executor:
            logger.error("BitvavoExecutor ikke initialisert, kan ikke hente live data.")
            return None
        since_timestamp = int(start_dt.timestamp() * 1000)
        target_end_timestamp = int(end_dt.timestamp() * 1000)
        limit_per_call = 500; all_ohlcv_list = []
        logger.info(f"Cache miss. Henter live data for {symbol} ({timeframe}) fra {start_dt.strftime('%Y-%m-%d')} til {end_dt.strftime('%Y-%m-%d')}")
        current_fetch_since_ts = since_timestamp
        while current_fetch_since_ts <= target_end_timestamp:
            try:
                chunk = await self.bitvavo_executor.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=current_fetch_since_ts, limit=limit_per_call)
                if not chunk: break
                last_valid_ts_in_chunk = -1
                for candle_data in chunk:
                    candle_ts = candle_data[0]
                    if candle_ts >= current_fetch_since_ts and candle_ts <= target_end_timestamp:
                        all_ohlcv_list.append(candle_data); last_valid_ts_in_chunk = candle_ts
                    elif candle_ts > target_end_timestamp: break
                if not chunk or last_valid_ts_in_chunk == -1:
                    last_candle_overall_ts = chunk[-1][0] if chunk else current_fetch_since_ts
                    if last_candle_overall_ts > target_end_timestamp: break
                    if chunk: current_fetch_since_ts = chunk[-1][0] + 1
                    else: break
                elif last_valid_ts_in_chunk >= target_end_timestamp: break
                else: current_fetch_since_ts = last_valid_ts_in_chunk + 1
                await asyncio.sleep(0.2)
            except Exception as e: logger.error(f"Fout under henting av live data chunk for {symbol} ({timeframe}) periode {start_dt}-{end_dt}: {e}"); return None
        if not all_ohlcv_list: logger.warning(f"Ingen OHLCV data hentet live for {symbol} ({timeframe}) for perioden {start_dt}-{end_dt}."); return None
        raw_df_for_caching = pd.DataFrame(all_ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        raw_df_for_caching['timestamp'] = raw_df_for_caching['timestamp'].astype('int64')
        for col in ['open', 'high', 'low', 'close', 'volume']: raw_df_for_caching[col] = pd.to_numeric(raw_df_for_caching[col], errors='coerce')
        self._write_to_cache(cache_filepath, raw_df_for_caching)
        return_df = raw_df_for_caching.copy(); return_df['date'] = pd.to_datetime(return_df['timestamp'], unit='ms')
        return_df.set_index('date', inplace=True)
        if 'timestamp' in return_df.columns: return_df.drop(columns=['timestamp'], inplace=True)
        return return_df

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
                        regime_start_dt = dt.strptime(period['start_date'], "%Y-%m-%d")
                        regime_end_dt = dt.strptime(period['end_date'], "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
                        if regime_start_dt > regime_end_dt:
                            logger.warning(f"Ongeldige periode in regimes voor {symbol} {regime_category}: start {period['start_date']} is na end {period['end_date']}. Overslaan.")
                            continue

                        period_df = await self._fetch_ohlcv_for_period(symbol, timeframe, regime_start_dt, regime_end_dt)
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
                start_dt_global = dt.strptime(global_start_date_str, "%Y-%m-%d")
                end_dt_global = dt.now()
                if global_end_date_str:
                    end_dt_global = dt.strptime(global_end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)

                if start_dt_global > end_dt_global:
                    logger.error(f"Global startdatum {global_start_date_str} is na sluttdato {global_end_date_str}.")
                    return {"all": pd.DataFrame()}

                global_df = await self._fetch_ohlcv_for_period(symbol, timeframe, start_dt_global, end_dt_global)
                if global_df is not None and not global_df.empty:
                    all_ohlcv_data_dfs_for_concatenation.append(global_df)
                    fetched_any_data = True
                    logger.info(f"Data ({len(global_df)} candles) gehaald voor {symbol}-{timeframe} via globale periode: {global_start_date_str} - {global_end_date_str if global_end_date_str else 'nu'}.")
                else:
                    logger.warning(f"Geen data gehaald voor {symbol}-{timeframe} met globale periode.")
            except ValueError as e:
                logger.error(f"Ugyldig globalt datoformat: {e}.")
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
                logger.info(f"Duplikaten verwijderd van geconcateneerde data. Resterende candles: {len(concatenated_df)}")
            concatenated_df.sort_index(inplace=True)
            concatenated_df.attrs['timeframe'] = timeframe
            concatenated_df.attrs['pair'] = symbol
            logger.info(f"Totaal {len(concatenated_df)} unieke candles voor 'all' data voor {symbol} ({timeframe}).")

        processed_regime_dfs["all"] = concatenated_df if concatenated_df is not None and not concatenated_df.empty else pd.DataFrame()
        if processed_regime_dfs["all"].empty:
            logger.warning(f"Uiteindelijk geen 'all' data beschikbaar voor {symbol} ({timeframe}).")


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
                regime_df.sort_index(inplace=True)
                regime_df.attrs['timeframe'] = timeframe
                regime_df.attrs['pair'] = symbol
                processed_regime_dfs[regime_cat] = regime_df
                logger.info(f"Data voor regime '{regime_cat}' ({symbol}-{timeframe}) verwerkt. {len(regime_df)} candles.")
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

        # Ensure technical indicators are calculated first (already seems to be the case)
        if not hasattr(self, 'cnn_pattern_detector'):
            from core.cnn_patterns import CNNPatterns
            self.cnn_pattern_detector = CNNPatterns()
            logger.info("CNNPatterns detector geïnitialiseerd in prepare_training_data.")
        if 'rsi' not in dataframe.columns: dataframe['rsi'] = ta.RSI(dataframe)
        if 'macd' not in dataframe.columns:
            macd_df = ta.MACD(dataframe)
            dataframe['macd'] = macd_df['macd']
            dataframe['macdsignal'] = macd_df['macdsignal']
            dataframe['macdhist'] = macd_df['macdhist']
        if 'bb_middleband' not in dataframe.columns:
            from freqtrade.vendor.qtpylib.indicators import bollinger_bands
            bollinger = bollinger_bands(ta.TYPPRICE(dataframe), window=20, stds=2)
            dataframe['bb_lowerband'] = bollinger['lower']
            dataframe['bb_middleband'] = bollinger['mid']
            dataframe['bb_upperband'] = bollinger['upper']

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

        if min_pattern_window_size > 0 and not dataframe['form_pattern_detected'].all():
            df_for_detection = dataframe.reset_index() # Requires 'date' or 'timestamp' column if index is DatetimeIndex
            if not isinstance(dataframe.index, pd.DatetimeIndex): # Should be DatetimeIndex
                 logger.error("DataFrame index is niet DatetimeIndex vóór form pattern detection loop.")
            # if 'date' not in df_for_detection.columns and dataframe.index.name == 'timestamp': # common after set_index
            #    df_for_detection.rename(columns={'timestamp':'date'}, inplace=True) # ensure 'date' exists if reset_index loses it

            for i in range(min_pattern_window_size - 1, len(df_for_detection)):
                # Ensure we are using original DataFrame's index for assignment
                original_df_index_at_i = dataframe.index[i]

                window_df_slice = df_for_detection.iloc[max(0, i - min_pattern_window_size + 1) : i + 1]
                if len(window_df_slice) < min_pattern_window_size: continue

                # _dataframe_to_candles expects 'open', 'high', 'low', 'close', 'volume', 'date'
                # Ensure 'date' column exists if it's not the index before this call, or adapt _dataframe_to_candles
                # Current _dataframe_to_candles uses .to_dict(orient='records'), so column names matter.
                # Assuming OHLCV are present. If 'date' is index, reset_index() would make it a column.
                # If index is already 'date', it should be fine.
                # If index is 'timestamp', reset_index() makes it a column.

                candle_list = self.cnn_pattern_detector._dataframe_to_candles(window_df_slice)
                detected = False
                if pattern_type == 'bullFlag': detected = self.cnn_pattern_detector.detect_bull_flag(candle_list)
                elif pattern_type == 'bearishEngulfing':
                    eng_type = self.cnn_pattern_detector._detect_engulfing(candle_list, "bearish")
                    if eng_type == "bearishEngulfing": detected = True

                if detected: dataframe.loc[original_df_index_at_i, 'form_pattern_detected'] = True

        form_detected_count = dataframe['form_pattern_detected'].sum()
        logger.info(f"For pattern '{pattern_type}' on {pair}-{tf}, {form_detected_count} samples initially identified by form detection.")

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

        # Cleanup
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
        # Ensure label_column_name exists, even if all labeling failed (it's initialized to 0)
        if label_column_name not in dataframe.columns: dataframe[label_column_name] = 0

        cols_to_drop_for_na = [label_column_name, 'future_high', 'future_low'] + feature_columns
        dataframe.dropna(subset=cols_to_drop_for_na, inplace=True)

        # Drop helper and intermediate columns
        columns_to_drop_finally = ['future_high', 'future_low', 'form_pattern_detected', 'gold_label_applied']
        dataframe.drop(columns=[col for col in columns_to_drop_finally if col in dataframe.columns], inplace=True, errors='ignore')

        logger.info(f"Trainingsdata voorbereid voor {pattern_type} ({pair}-{tf}) met {len(dataframe)} samples. Label: {label_column_name}, Positives: {dataframe[label_column_name].sum()}")
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
            model_save_path = os.path.join(model_dir, f'cnn_model_{pattern_type}_{current_arch_key_orig}_hpo_{regime_name}.pth')
            scaler_save_path = os.path.join(model_dir, f'scaler_params_{pattern_type}_{current_arch_key_orig}_hpo_{regime_name}.json')
            logger.info(f"HPO uitgevoerd. Modelnaam prefix: {model_name_prefix}")
        else:
            # No HPO, use regime name in the prefix and paths
            model_name_prefix = f"{model_name_base}_{regime_name}"
            model_save_path = os.path.join(model_dir, f'cnn_model_{pattern_type}_{current_arch_key_orig}_{regime_name}.pth')
            scaler_save_path = os.path.join(model_dir, f'scaler_params_{pattern_type}_{current_arch_key_orig}_{regime_name}.json')
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
                torch.save(final_model.state_dict(), model_save_path)
                logger.debug(f"Beste final model '{model_name_prefix}' opgeslagen (Epoch {epoch+1}) naar {model_save_path}. Val Loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience: logger.info(f"Early stopping final model '{model_name_prefix}' (Epoch {epoch+1})."); break
        try:
            if os.path.exists(model_save_path): logger.info(f"'{model_name_prefix}' final training voltooid. Model opgeslagen: {model_save_path}")
            else: logger.warning(f"'{model_name_prefix}' final training voltooid. Geen model opgeslagen: {model_save_path}.")
            scaler_params_to_save = {'feature_names_in_':feature_names_in_, 'min_':min_, 'scale_':scale_, 'sequence_length':sequence_length}
            with open(scaler_save_path, 'w') as f: json.dump(scaler_params_to_save, f, indent=4)
            logger.info(f"Scaler params '{model_name_prefix}' opgeslagen: {scaler_save_path}")
        except Exception as e: logger.error(f"Fout opslaan final model/scaler '{model_name_prefix}': {e}")

        await asyncio.to_thread(self._log_pretrain_activity, model_type=model_name_prefix, regime_name=regime_name, data_size=len(X_train_full)+len(X_val_full),
                                model_path_saved=model_save_path, scaler_params_path_saved=scaler_save_path,
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
            if os.path.exists(PRETRAIN_LOG_FILE) and os.path.getsize(PRETRAIN_LOG_FILE) > 0:
                with open(PRETRAIN_LOG_FILE, 'r', encoding='utf-8') as f: logs = json.load(f)
                if not isinstance(logs, list): logs = [logs]
            logs.append(entry)
            with open(PRETRAIN_LOG_FILE, 'w', encoding='utf-8') as f: json.dump(logs, f, indent=2)
        except Exception as e: logger.error(f"Fout bij loggen pre-train activiteit: {e}")

    async def analyze_time_of_day_effectiveness(self, historical_data: pd.DataFrame, strategy_id: str) -> Dict[str, Any]:
        # ... (implementation as before) ...
        logger.info(f"Analyseren tijd-van-dag effectiviteit voor strategie {strategy_id}...")
        if historical_data.empty: return {}
        label_to_analyze = next((lbl for lbl in ['bullFlag_label', 'bearishEngulfing_label'] if lbl in historical_data.columns), None)
        if not label_to_analyze:
            label_to_analyze = next((col for col in historical_data.columns if col.endswith('_label')), None)
            if label_to_analyze: logger.info(f"Generiek label '{label_to_analyze}' gevonden voor tijd-van-dag analyse.")
            else: logger.warning("Geen label kolom (*_label) gevonden voor tijd-van-dag analyse."); return {}
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            logger.error("DataFrame index geen DatetimeIndex. Kan uur niet extraheren."); return {}
        df_copy = historical_data.copy(); df_copy['hour_of_day'] = df_copy.index.hour
        time_effectiveness = df_copy.groupby('hour_of_day')[label_to_analyze].agg(['mean', 'count']).rename(columns={'mean': f'avg_{label_to_analyze}_proportion', 'count': 'num_samples'})
        result_dict = time_effectiveness.to_dict(orient='index')
        logger.info(f"Tijd-van-dag effectiviteit (obv {label_to_analyze}) geanalyseerd: {result_dict}")
        try:
            with open(TIME_EFFECTIVENESS_FILE, 'w', encoding='utf-8') as f: json.dump(result_dict, f, indent=2)
        except Exception as e: logger.error(f"Fout bij opslaan tijd-van-dag effectiviteit: {e}")
        return result_dict

    async def run_pretraining_pipeline(self, strategy_id: str, params_manager=None):
        # ... (implementation as before) ...
        logger.info(f"Start pre-training pipeline voor strategie: {strategy_id}...")
        if params_manager: self.params_manager = params_manager
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

        # --- Start Backtesting Phase ---
        perform_backtesting = self.params_manager.get_param('perform_backtesting', False)
        if perform_backtesting and self.backtester:
            logger.info(f"--- Starten Backtesting Fase voor strategie: {strategy_id} ---")
            if len(regimes_to_train_for) > 1 or "all" not in regimes_to_train_for:
                 logger.warning("Meerdere regimes zijn getraind. Backtesting gebruikt momenteel modellen die getraind zijn op 'all' data (of de standaard modelnaam zonder regime suffix, of met _hpo_all).")

            sequence_length_cnn = self.params_manager.get_param('sequence_length_cnn', 30)
            # Determine architecture key for backtesting. If HPO was run for 'all' regime, it might have a specific name.
            # This assumes HPO for "all" regime would result in parameters used by current_cnn_architecture_key or a suffixed version.
            current_arch_key_for_bt = self.params_manager.get_param('current_cnn_architecture_key', "default_simple")
            # If HPO was generally performed (e.g. for the 'all' regime), backtester should try to load the HPO version of the 'all' model.
            # The backtester's model loading logic might need to be aware of the _hpo_all suffix.
            # For now, we pass the base architecture key, and if an HPO'd 'all' model exists with a conventional name, backtester should find it.

            for pattern_type_bt in patterns_to_train_list:
                logger.info(f"--- Backtesting voor patroontype: {pattern_type_bt} ---")
                for pair_bt in pairs_to_fetch:
                    for timeframe_bt in timeframes_to_fetch:
                        # The architecture_key passed to backtester should ideally point to the model trained on "all" data,
                        # potentially an HPO version if HPO was run for the "all" regime.
                        # The backtester would need logic to find e.g. {arch_key}_hpo_all.pth then {arch_key}_all.pth.
                        # For this subtask, we'll pass the base arch key and assume "all" regime.
                        logger.info(f"--- Backtesten {pair_bt} ({timeframe_bt}) voor patroon: {pattern_type_bt}, architectuur: {current_arch_key_for_bt} (veronderstelt 'all' regime model) ---")
                        await self.backtester.run_backtest(
                            symbol=pair_bt,
                            timeframe=timeframe_bt,
                            pattern_type=pattern_type_bt,
                            architecture_key=current_arch_key_for_bt,
                            sequence_length=sequence_length_cnn,
                            regime_filter="all" # Explicitly pass "all" to ensure backtester loads the correct model if it's regime-aware
                        )
            logger.info(f"--- Backtesting Fase voltooid voor strategie {strategy_id} (gebruikmakend van 'all' data modellen) ---")
        elif not self.backtester:
            logger.warning("Backtester object is niet geïnitialiseerd. Backtesting overgeslagen.")
        else:
            logger.info("Perform_backtesting is False in parameters. Backtesting overgeslagen.")

        logger.info(f"Pre-training en backtesting pipeline voltooid voor strategie {strategy_id} for alle konfigurerte patterns.")

if __name__ == "__main__":
    import sys # Ensure sys is imported for logging handler
    from pathlib import Path # For Path object usage in __main__
    dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    from core.params_manager import ParamsManager
    from core.cnn_patterns import CNNPatterns

    BITVAVO_API_KEY = os.getenv('BITVAVO_API_KEY')
    BITVAVO_SECRET_KEY = os.getenv('BITVAVO_SECRET_KEY')
    if not BITVAVO_API_KEY or not BITVAVO_SECRET_KEY:
        logger.warning("Bitvavo API keys niet gevonden in .env. Data ophalen zal waarschijnlijk falen.")

    async def run_test_pre_trainer():
        test_pairs_for_test = ["ETH/EUR"]
        test_timeframe = "1h"
        test_patterns_to_train = ["bullFlag"]
        test_arch_key = "default_simple" # Using a fixed arch key for test predictability
        test_regimes = ["all", "testbull"]

        # Define test-specific paths
        test_data_base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / '..' / 'data' / 'test_pretrainer_artifacts'
        test_gold_standard_dir = test_data_base_dir / 'gold_standard_data'
        test_market_regimes_file = test_data_base_dir / 'market_regimes_test.json' # Different from main one

        logger.info(f"--- STARTING INTEGRATION TEST FOR PRETRAINER (ALL FEATURES) ---")
        logger.info(f"Test config: Pair={test_pairs_for_test[0]}, TF={test_timeframe}, Pattern={test_patterns_to_train[0]}, Arch={test_arch_key}, Regimes={test_regimes}")
        logger.info(f"Test artifacts base dir: {test_data_base_dir}")

        # --- Artifact Clearing (before test run) ---
        # Clear model outputs for this specific test config
        for pair_name in test_pairs_for_test:
            pair_sani = pair_name.replace('/', '_')
            model_output_dir = Path(MODELS_DIR) / pair_sani / test_timeframe
            if model_output_dir.exists(): shutil.rmtree(model_output_dir)

        # Clear general test artifacts dir
        if test_data_base_dir.exists(): shutil.rmtree(test_data_base_dir)
        os.makedirs(test_gold_standard_dir, exist_ok=True)

        # Clear main log files that PreTrainer uses
        if os.path.exists(PRETRAIN_LOG_FILE): os.remove(PRETRAIN_LOG_FILE)
        if os.path.exists(TIME_EFFECTIVENESS_FILE): os.remove(TIME_EFFECTIVENESS_FILE)
        if os.path.exists(MARKET_REGIMES_FILE): os.remove(MARKET_REGIMES_FILE) # Remove main one if it exists, test will create its own

        # Clear backtest results file
        backtest_results_path = Path(MEMORY_DIR) / 'backtest_results.json'
        if backtest_results_path.exists(): os.remove(backtest_results_path)

        logger.info(f"--- Test Setup: Artifacts Cleared ---")

        # --- Create Dummy Gold Standard CSV ---
        dummy_gold_csv_path = test_gold_standard_dir / f"{test_pairs_for_test[0].replace('/', '_')}_{test_timeframe}_{test_patterns_to_train[0]}_gold.csv"
        with open(dummy_gold_csv_path, 'w') as f:
            f.write("timestamp,open,high,low,close,volume,gold_label\n")
            # Timestamps for Nov 2, 2023, 00:00 GMT and 01:00 GMT (ensure these are within test data fetch range)
            f.write("1698883200000,1800,1801,1799,1800.5,10,1\n") # Nov 2, 00:00
            f.write("1698886800000,1800.5,1802,1799.5,1801.0,12,0\n") # Nov 2, 01:00
        logger.info(f"Dummy gold standard CSV created at: {dummy_gold_csv_path}")

        # --- Create Dummy Market Regimes JSON ---
        # This file needs to be at the location PreTrainer expects (MARKET_REGIMES_FILE)
        # So we overwrite the main path for the duration of this test or use a param to point to test file.
        # For now, let's make PreTrainer use our test_market_regimes_file by setting its path in params.
        # However, PreTrainer reads MARKET_REGIMES_FILE directly. So, we create it at that location.
        dummy_regimes_content = {
            test_pairs_for_test[0]: {
                "testbull": [{"start_date": "2023-11-01", "end_date": "2023-11-03"}], # Short period for testing
                "testbear": [{"start_date": "2023-11-04", "end_date": "2023-11-06"}]  # Another regime
            }
        }
        with open(MARKET_REGIMES_FILE, 'w') as f: # Overwrite the main file path for the test
            json.dump(dummy_regimes_content, f, indent=4)
        logger.info(f"Dummy market regimes JSON created at: {MARKET_REGIMES_FILE}")


        # --- ParamsManager Setup for Test ---
        params_manager_instance = ParamsManager()
        # Data fetch range should include gold standard and regime dates
        await params_manager_instance.set_param("data_fetch_start_date_str", "2023-10-30")
        await params_manager_instance.set_param("data_fetch_end_date_str", "2023-11-10")
        await params_manager_instance.set_param("data_fetch_pairs", test_pairs_for_test)
        await params_manager_instance.set_param("data_fetch_timeframes", [test_timeframe])
        await params_manager_instance.set_param("patterns_to_train", test_patterns_to_train)

        await params_manager_instance.set_param('sequence_length_cnn', 5) # Smaller for faster test
        await params_manager_instance.set_param('num_epochs_cnn', 1)
        await params_manager_instance.set_param('batch_size_cnn', 4)
        await params_manager_instance.set_param('num_epochs_cnn_hpo_trial', 1) # Epochs for HPO trials

        # Gold Standard
        await params_manager_instance.set_param("gold_standard_data_path", str(test_gold_standard_dir))

        # HPO
        await params_manager_instance.set_param('perform_hyperparameter_optimization', True)
        await params_manager_instance.set_param('hpo_num_trials', 2) # Minimal trials
        await params_manager_instance.set_param('hpo_timeout_seconds', 60) # Short timeout

        # Regimes
        await params_manager_instance.set_param('regimes_to_train', test_regimes) # Test 'all' and 'testbull'

        # Labeling config (ensure it's simple for testing)
        current_configs = params_manager_instance.get_param('pattern_labeling_configs')
        if not current_configs: current_configs = {}
        current_configs.setdefault(test_patterns_to_train[0], {}).update({ "future_N_candles": 3, "profit_threshold_pct": 0.001, "loss_threshold_pct": -0.001 })
        await params_manager_instance.set_param('pattern_labeling_configs', current_configs)

        await params_manager_instance.set_param('current_cnn_architecture_key', test_arch_key)
        # Ensure the test_arch_key exists in cnn_architecture_configs or HPO will use its own suggestions
        cnn_arch_configs = params_manager_instance.get_param('cnn_architecture_configs')
        if test_arch_key not in cnn_arch_configs:
            cnn_arch_configs[test_arch_key] = {"num_conv_layers": 1, "filters_per_layer": [16], "kernel_sizes_per_layer": [3], "strides_per_layer": [1], "padding_per_layer": [1], "pooling_types_per_layer": ["max"], "pooling_kernel_sizes_per_layer": [2], "pooling_strides_per_layer": [2], "use_batch_norm": False, "dropout_rate": 0.1}
            await params_manager_instance.set_param('cnn_architecture_configs', cnn_arch_configs)

        logger.info(f"TEST SETUP: ParamsManager configured for Gold Standard, HPO, and Regimes.")

        # --- PreTrainer Execution ---
        pre_trainer = PreTrainer(params_manager=params_manager_instance)
        test_strategy_id = "TestStrategy_FullFeatures"

        try:
            print(f"\n--- EXECUTING PreTrainer Pipeline (Gold, HPO, Regimes, CV, Backtesting) ---")
            await pre_trainer.run_pretraining_pipeline(strategy_id=test_strategy_id, params_manager=params_manager_instance)
        finally:
            # --- Cleanup Test-Specific Artifacts ---
            logger.info(f"--- Cleaning up test-specific artifacts ---")
            if test_data_base_dir.exists():
                shutil.rmtree(test_data_base_dir)
                logger.info(f"Removed test artifacts base directory: {test_data_base_dir}")
            # Remove the market_regimes.json created at the main path
            if os.path.exists(MARKET_REGIMES_FILE):
                 os.remove(MARKET_REGIMES_FILE)
                 logger.info(f"Removed dummy market regimes file: {MARKET_REGIMES_FILE}")


        # --- Validation of Outputs ---
        print(f"\n--- VALIDATION OF OUTPUTS (Gold, HPO, Regimes) ---")

        # Validate Model and Scaler files for each regime trained
        for pair_name in test_pairs_for_test:
            pair_sani = pair_name.replace('/', '_')
            for pattern in test_patterns_to_train:
                for regime_name_val in test_regimes: # test_regimes = ["all", "testbull"]
                    model_dir_val = Path(MODELS_DIR) / pair_sani / test_timeframe
                    # Filename includes _hpo_ because HPO is enabled
                    expected_model_fname = f'cnn_model_{pattern}_{test_arch_key}_hpo_{regime_name_val}.pth'
                    expected_scaler_fname = f'scaler_params_{pattern}_{test_arch_key}_hpo_{regime_name_val}.json'

                    expected_model_file = model_dir_val / expected_model_fname
                    expected_scaler_file = model_dir_val / expected_scaler_fname

                    assert expected_model_file.exists(), f"Model file MISSING: {expected_model_file}"
                    print(f"SUCCESS: Model file for {regime_name_val} regime created: {expected_model_file.name}")
                    assert expected_scaler_file.exists(), f"Scaler file MISSING: {expected_scaler_file}"
                    print(f"SUCCESS: Scaler file for {regime_name_val} regime created: {expected_scaler_file.name}")

        # Validate Pretrain Log for HPO and Regime entries
        assert os.path.exists(PRETRAIN_LOG_FILE), f"Pretrain log file MISSING: {PRETRAIN_LOG_FILE}"
        with open(PRETRAIN_LOG_FILE, 'r') as f:
            log_entries = json.load(f)

        assert isinstance(log_entries, list), "Pretrain log is not a list."
        # Expecting entries for each regime (all, testbull)
        # Since CV is also on, if it runs before HPO it might create entries too.
        # For this test, HPO runs first within train_ai_models, then CV (if enabled, it is).
        # The log entries from HPO trials are not stored here, only final model training log.

        found_all_regime_log = False
        found_testbull_regime_log = False
        hpo_params_found_in_log = False

        for entry in log_entries:
            assert "model_type" in entry
            assert "regime_name" in entry
            assert "cross_validation_results" in entry # CV is on

            if entry["regime_name"] == "all" and f"{test_arch_key}_hpo_all" in entry["model_type"]:
                found_all_regime_log = True
                if "best_trial_params" in entry.get("cross_validation_results", {}).get("hpo_results", {}): # Assuming HPO results get logged under CV for now
                     hpo_params_found_in_log = True
            if entry["regime_name"] == "testbull" and f"{test_arch_key}_hpo_testbull" in entry["model_type"]:
                found_testbull_regime_log = True

        assert found_all_regime_log, "Log entry for 'all' regime (with HPO suffix) not found."
        assert found_testbull_regime_log, "Log entry for 'testbull' regime (with HPO suffix) not found."
        # The HPO params themselves are in study.best_trial.params logged during HPO, not directly in _log_pretrain_activity's main params.
        # _log_pretrain_activity could be enhanced to store best_trial.params. For now, checking model name is key.
        print("SUCCESS: Pretrain log contains entries for 'all' and 'testbull' regimes with HPO naming.")

        # Validate Backtest Results File (Backtesting runs on "all" model by default)
        assert os.path.exists(backtest_results_path), f"Backtest results file MISSING: {backtest_results_path}"
            with open(backtest_results_path, 'r') as f:
                try:
                    backtest_log_entries = json.load(f)
                    assert isinstance(backtest_log_entries, list) and len(backtest_log_entries) > 0, "Backtest log is empty or not a list."
                    print(f"SUCCESS: Backtest results file created and contains {len(backtest_log_entries)} entries.")
                    # Further checks on backtest content can be added if needed, e.g., ensuring it used the HPO "all" model.
                    # For now, existence and basic format is checked.
                    first_bt_entry = backtest_log_entries[0]
                    assert first_bt_entry['architecture_key'] == test_arch_key, f"Backtest ran with {first_bt_entry['architecture_key']} instead of {test_arch_key}"
                    # If HPO was run for 'all' regime, the backtester might use a model like 'default_simple_hpo_all'
                    # This depends on how Backtester.load_model constructs the filename.
                    # Current test setup for backtesting in PreTrainer passes `current_arch_key_for_bt` (which is `test_arch_key`)
                    # and `regime_filter="all"`. The Backtester needs to correctly interpret this to find the HPO-All model.
                    # For now, we assume it uses the base key and the backtester's loading logic handles finding the HPO version.
                    print(f"Backtest entry seems OK, used architecture key: {first_bt_entry['architecture_key']}")

                except json.JSONDecodeError:
                    assert False, f"FAILURE: Could not decode JSON from backtest results file {backtest_results_path}."
        else:
            assert False, f"FAILURE: Backtest results file {backtest_results_path} not found."


        # --- Optional: Test SimpleCNN Custom Architecture Instantiation (already present, keep) ---
        print("\n--- Testing SimpleCNN Custom Architecture Instantiation (original test) ---")
        try:
            num_input_features_example = 12
            custom_cnn = SimpleCNN( input_channels=num_input_features_example, num_classes=2, sequence_length=30,
                num_conv_layers=3, filters_per_layer=[16,32,64], kernel_sizes_per_layer=[3,3,5],
                strides_per_layer=[1,1,1], padding_per_layer=[1,1,2],
                pooling_types_per_layer=['max','max',None], pooling_kernel_sizes_per_layer=[2,2,1],
                pooling_strides_per_layer=[2,2,1], use_batch_norm=True, dropout_rate=0.25)
            print("SUCCESS: Custom SimpleCNN instantiated.")
        except Exception as e: print(f"FAILURE: Custom SimpleCNN instantiation failed: {e}"); logger.error(f"Custom SimpleCNN instantiation failed: {e}", exc_info=True)

        logger.info("--- INTEGRATION TEST FOR PRETRAINER (CV & BACKTESTING) COMPLETED ---")

    asyncio.run(run_test_pre_trainer())
