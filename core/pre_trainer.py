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

    async def fetch_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # ... (implementation as before) ...
        if not self.bitvavo_executor:
            logger.error("BitvavoExecutor niet geïnitialiseerd. Kan geen historische data ophalen.")
            return pd.DataFrame()
        all_ohlcv_data_dfs = []
        fetched_via_regimes = False
        if symbol in self.market_regimes and self.market_regimes[symbol]:
            logger.info(f"Marktregimes gevonden voor {symbol}. Data wordt per regime opgehaald.")
            symbol_regimes = self.market_regimes[symbol]
            for regime_category, periods in symbol_regimes.items():
                if not periods: continue
                logger.info(f"Verwerken regime categorie: {regime_category} voor {symbol}")
                for period in periods:
                    try:
                        regime_start_dt = dt.strptime(period['start_date'], "%Y-%m-%d")
                        regime_end_dt = dt.strptime(period['end_date'], "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
                        if regime_start_dt > regime_end_dt:
                            logger.warning(f"Ongeldige periode i regimes for {symbol} {regime_category}: start {period['start_date']} er etter end {period['end_date']}. Hopper over.")
                            continue
                        period_df = await self._fetch_ohlcv_for_period(symbol, timeframe, regime_start_dt, regime_end_dt)
                        if period_df is not None and not period_df.empty:
                            all_ohlcv_data_dfs.append(period_df); fetched_via_regimes = True
                    except ValueError as e: logger.error(f"Ugyldig datoformat i regimes for {symbol} {regime_category} periode {period}: {e}. Hopper over.")
                    except Exception as e: logger.error(f"Generell feil ved behandling av regime {symbol} {regime_category} periode {period}: {e}. Hopper over.")
            if fetched_via_regimes and all_ohlcv_data_dfs: logger.info(f"Data for {symbol} ({timeframe}) hentet via markedsregimer.")
            elif fetched_via_regimes: logger.warning(f"Regimer definert for {symbol} ({timeframe}) men ingen data hentet.")
        if not fetched_via_regimes:
            logger.info(f"Ingen spesifikke regimer (med data) funnet/behandlet for {symbol} ({timeframe}), eller ingen regimer definert. Global datoperiode brukes.")
            global_start_date_str = self.params_manager.get_param("data_fetch_start_date_str")
            global_end_date_str = self.params_manager.get_param("data_fetch_end_date_str")
            if not global_start_date_str: logger.error(f"Ingen 'data_fetch_start_date_str' funnet i ParamsManager for {symbol} ({timeframe})."); return pd.DataFrame()
            try:
                start_dt_global = dt.strptime(global_start_date_str, "%Y-%m-%d")
                end_dt_global = dt.now()
                if global_end_date_str: end_dt_global = dt.strptime(global_end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999)
                if start_dt_global > end_dt_global: logger.error(f"Global startdato {global_start_date_str} er etter sluttdato {global_end_date_str}."); return pd.DataFrame()
                global_df = await self._fetch_ohlcv_for_period(symbol, timeframe, start_dt_global, end_dt_global)
                if global_df is not None and not global_df.empty:
                     all_ohlcv_data_dfs.append(global_df)
                     logger.info(f"Data for {symbol} ({timeframe}) hentet via global periode: {global_start_date_str} - {global_end_date_str if global_end_date_str else 'nåtid'}.")
                else: logger.warning(f"Ingen data hentet for {symbol} ({timeframe}) med global periode.")
            except ValueError as e: logger.error(f"Ugyldig globalt datoformat: {e}."); return pd.DataFrame()
        if not all_ohlcv_data_dfs: logger.warning(f"Til slutt ingen historisk data tilgjengelig for {symbol} ({timeframe})."); return pd.DataFrame()
        final_df = pd.concat(all_ohlcv_data_dfs, ignore_index=False)
        if final_df.empty: logger.warning(f"Tom DataFrame etter concat for {symbol} ({timeframe})."); return pd.DataFrame()
        if not isinstance(final_df.index, pd.DatetimeIndex):
             logger.error(f"Index for {symbol} ({timeframe}) er ikke DatetimeIndex etter concat.")
             if 'date' in final_df.columns: final_df['date'] = pd.to_datetime(final_df['date']); final_df.set_index('date', inplace=True)
             else: return pd.DataFrame()
        if not final_df.index.is_unique: final_df = final_df[~final_df.index.duplicated(keep='first')]; logger.info(f"Duplikater fjernet. Resterende candles: {len(final_df)}")
        final_df.sort_index(inplace=True)
        final_df.attrs['timeframe'] = timeframe; final_df.attrs['pair'] = symbol
        logger.info(f"Totalt {len(final_df)} unike candles forberedt for {symbol} ({timeframe}). Fetched via regimes: {fetched_via_regimes}")
        return final_df

    async def prepare_training_data(self, dataframe: pd.DataFrame, pattern_type: str) -> pd.DataFrame:
        # ... (implementation as before, including form detection and new logging) ...
        pair = dataframe.attrs.get('pair', 'N/A'); tf = dataframe.attrs.get('timeframe', 'N/A')
        logger.info(f"Voorbereiden trainingsdata voor {pair} ({tf}), pattern: {pattern_type}...")
        if dataframe.empty: logger.warning(f"Lege dataframe ontvangen in prepare_training_data for {pattern_type}."); return dataframe
        if not hasattr(self, 'cnn_pattern_detector'):
            from core.cnn_patterns import CNNPatterns
            self.cnn_pattern_detector = CNNPatterns(); logger.info("CNNPatterns detector geïnitialiseerd in prepare_training_data.")
        if 'rsi' not in dataframe.columns: dataframe['rsi'] = ta.RSI(dataframe)
        if 'macd' not in dataframe.columns:
            macd_df = ta.MACD(dataframe); dataframe['macd'] = macd_df['macd']
            dataframe['macdsignal'] = macd_df['macdsignal']; dataframe['macdhist'] = macd_df['macdhist']
        if 'bb_middleband' not in dataframe.columns:
            from freqtrade.vendor.qtpylib.indicators import bollinger_bands
            bollinger = bollinger_bands(ta.TYPPRICE(dataframe), window=20, stds=2)
            dataframe['bb_lowerband'] = bollinger['lower']; dataframe['bb_middleband'] = bollinger['mid']; dataframe['bb_upperband'] = bollinger['upper']
        dataframe['form_pattern_detected'] = False; min_pattern_window_size = 0
        if pattern_type == 'bullFlag': min_pattern_window_size = 10
        elif pattern_type == 'bearishEngulfing': min_pattern_window_size = 2
        else: logger.warning(f"Form detection not implemented for '{pattern_type}' for {pair}-{tf}. Defaulting all to True."); dataframe['form_pattern_detected'] = True
        if min_pattern_window_size > 0 and not dataframe['form_pattern_detected'].all():
            df_for_detection = dataframe.reset_index()
            for i in range(min_pattern_window_size - 1, len(df_for_detection)):
                if i >= len(dataframe.index): continue
                window_df_slice = df_for_detection.iloc[max(0, i - min_pattern_window_size + 1) : i + 1]
                if len(window_df_slice) < min_pattern_window_size: continue
                candle_list = self.cnn_pattern_detector._dataframe_to_candles(window_df_slice)
                detected = False
                if pattern_type == 'bullFlag': detected = self.cnn_pattern_detector.detect_bull_flag(candle_list)
                elif pattern_type == 'bearishEngulfing':
                    eng_type = self.cnn_pattern_detector._detect_engulfing(candle_list, "bearish")
                    if eng_type == "bearishEngulfing": detected = True
                if detected: dataframe.loc[dataframe.index[i], 'form_pattern_detected'] = True
        form_detected_count = dataframe['form_pattern_detected'].sum()
        logger.info(f"For pattern '{pattern_type}' on {pair}-{tf}, {form_detected_count} samples initially identified by form detection.")
        label_column_name = f"{pattern_type}_label"
        configs = self.params_manager.get_param('pattern_labeling_configs')
        if not configs:
            logger.error("`pattern_labeling_configs` niet gevonden. Labeling gestopt."); dataframe[label_column_name] = 0
            if 'form_pattern_detected' in dataframe.columns: dataframe.drop(columns=['form_pattern_detected'], inplace=True, errors='ignore')
            return dataframe
        cfg = configs.get(pattern_type)
        if not cfg:
            logger.warning(f"Config for '{pattern_type}' niet gevonden. Labeling overgeslagen, kolom '{label_column_name}' op 0 gezet.")
            dataframe[label_column_name] = 0
            if 'form_pattern_detected' in dataframe.columns: dataframe.drop(columns=['form_pattern_detected'], inplace=True, errors='ignore')
            return dataframe
        future_N, profit_thresh, loss_thresh = cfg.get('future_N_candles',20), cfg.get('profit_threshold_pct',0.02), cfg.get('loss_threshold_pct',-0.01)
        dataframe['future_high'] = dataframe['high'].shift(-future_N); dataframe['future_low'] = dataframe['low'].shift(-future_N)
        dataframe[label_column_name] = 0
        if form_detected_count > 0:
            detected_indices = dataframe.index[dataframe['form_pattern_detected']]
            if pattern_type == 'bullFlag':
                profit_met = (dataframe.loc[detected_indices, 'future_high'] - dataframe.loc[detected_indices, 'close']) / dataframe.loc[detected_indices, 'close'] >= profit_thresh
                loss_met = (dataframe.loc[detected_indices, 'future_low'] - dataframe.loc[detected_indices, 'close']) / dataframe.loc[detected_indices, 'close'] <= loss_thresh
                dataframe.loc[detected_indices, label_column_name] = np.where(profit_met & ~loss_met, 1, 0)
            elif pattern_type == 'bearishEngulfing':
                profit_met = (dataframe.loc[detected_indices, 'close'] - dataframe.loc[detected_indices, 'future_low']) / dataframe.loc[detected_indices, 'close'] >= profit_thresh
                loss_met = (dataframe.loc[detected_indices, 'future_high'] - dataframe.loc[detected_indices, 'close']) / dataframe.loc[detected_indices, 'close'] >= abs(loss_thresh)
                dataframe.loc[detected_indices, label_column_name] = np.where(profit_met & ~loss_met, 1, 0)
        positive_labels_count = dataframe[label_column_name].sum()
        logger.info(f"For pattern '{pattern_type}' on {pair}-{tf}, out of {form_detected_count} form-identified samples, {positive_labels_count} were ultimately labeled as positive (1).")
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
        if label_column_name not in dataframe.columns: dataframe[label_column_name] = 0
        cols_to_drop_for_na = [label_column_name, 'future_high', 'future_low'] + feature_columns
        dataframe.dropna(subset=cols_to_drop_for_na, inplace=True)
        dataframe.drop(columns=['future_high', 'future_low', 'form_pattern_detected'], inplace=True, errors='ignore')
        logger.info(f"Trainingsdata voorbereid voor {pattern_type} ({pair}-{tf}) met {len(dataframe)} samples. Label: {label_column_name}, Positives: {dataframe[label_column_name].sum()}")
        return dataframe

    async def train_ai_models(self, training_data: pd.DataFrame, symbol: str, timeframe: str, pattern_type: str, target_label_column: str):
        current_arch_key = self.params_manager.get_param('current_cnn_architecture_key', "default_simple")
        all_arch_configs = self.params_manager.get_param('cnn_architecture_configs')
        arch_params = {}
        if all_arch_configs and current_arch_key in all_arch_configs:
            arch_params = all_arch_configs[current_arch_key]
            logger.info(f"Gebruikt CNN architectuur '{current_arch_key}' voor training.")
        else: # Fallback to SimpleCNN defaults if specific arch_key not found or configs are missing
            logger.warning(f"CNN architectuur '{current_arch_key}' niet gevonden in ParamsManager of cnn_architecture_configs is None. Fallback naar SimpleCNN defaults (impliciet 'default_simple' equivalent).")
            current_arch_key = "default_simple" # Ensure key reflects fallback for naming consistency

        model_name_prefix = f"{symbol.replace('/', '_')}_{timeframe}_{pattern_type}_{current_arch_key}"
        logger.info(f"Start training PyTorch AI-model '{model_name_prefix}' met {len(training_data)} samples...")

        if training_data.empty: logger.warning(f"Geen trainingsdata voor '{model_name_prefix}'."); return

        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
        if not all(col in training_data.columns for col in feature_columns + [target_label_column]):
            missing_cols = [col for col in feature_columns + [target_label_column] if col not in training_data.columns]
            logger.error(f"Benodigde kolommen ({missing_cols}) niet aanwezig in training_data voor '{model_name_prefix}'. Overslaan.")
            return

        symbol_underscore = symbol.replace('/', '_')
        model_dir = os.path.join(MODELS_DIR, symbol_underscore, timeframe)
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f'cnn_model_{pattern_type}_{current_arch_key}.pth')
        scaler_save_path = os.path.join(model_dir, f'scaler_params_{pattern_type}_{current_arch_key}.json')

        sequence_length = self.params_manager.get_param('sequence_length_cnn', 30)
        num_epochs = self.params_manager.get_param('num_epochs_cnn', 10)
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


        # --- Finaal model trainen op de volledige X_train_full, y_train_full ---
        logger.info(f"Starten van definitieve modeltraining voor {model_name_prefix} op volledige trainingsset...")
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
                for data, targets in val_loader_full: # Valideer op X_val_full, y_val_full
                    outputs = final_model(data); current_epoch_val_loss += final_criterion(outputs, targets).item() * data.size(0)
                    all_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy()) # Store probabilities
                    all_targets.extend(targets.cpu().numpy())

            avg_epoch_val_loss = current_epoch_val_loss / len(val_dataset_full) if len(val_dataset_full) > 0 else float('nan')
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

        await asyncio.to_thread(self._log_pretrain_activity, model_type=model_name_prefix, data_size=len(X_train_full)+len(X_val_full),
                                model_path_saved=model_save_path, scaler_params_path_saved=scaler_save_path,
                                cv_results=cv_results, **best_metrics) # Pass CV results here
        logger.info(f"Pre-training {model_name_prefix} voltooid. Beste Val Loss: {best_metrics.get('loss', float('inf')):.4f}, Acc: {best_metrics.get('accuracy', 0.0):.4f}")
        return best_metrics.get('loss')

    def _log_pretrain_activity(self, model_type: str, data_size: int, model_path_saved: str, scaler_params_path_saved: str,
                             best_val_loss: float = None, best_val_accuracy: float = None,
                             best_val_precision: float = None, best_val_recall: float = None,
                             best_val_f1: float = None, auc: float = None, # Added AUC from best_metrics
                             cv_results: Optional[dict] = None): # Added cv_results
        entry = {"timestamp": datetime.now().isoformat(), "model_type": model_type, "data_size": data_size,
                 "status": "completed_pytorch_training", "model_path_saved": model_path_saved,
                 "scaler_params_path_saved": scaler_params_path_saved,
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
        if not pairs_to_fetch or not timeframes_to_fetch: logger.error("Geen 'data_fetch_pairs' of 'data_fetch_timeframes' in params. Pipeline gestopt."); return
        if not patterns_to_train_list: logger.warning("Ingen 'patterns_to_train' funnet i parametere. Standard til ['bullFlag']."); patterns_to_train_list = ["bullFlag"]
        logger.info(f"Pipeline voor paren: {pairs_to_fetch}, timeframes: {timeframes_to_fetch}, patterns: {patterns_to_train_list}")
        for pattern_type in patterns_to_train_list:
            logger.info(f"--- Starter treningsrunde for patrontype: {pattern_type} ---")
            for pair in pairs_to_fetch:
                for timeframe in timeframes_to_fetch:
                    logger.info(f"--- Verwerken {pair} ({timeframe}) voor patroon: {pattern_type} ---")
                    historical_df = await self.fetch_historical_data(pair, timeframe)
                    if historical_df.empty: logger.error(f"Geen hist. data voor {pair} ({timeframe}). Hopper over."); continue
                    processed_df = await self.prepare_training_data(historical_df.copy(), pattern_type=pattern_type)
                    if processed_df.empty: logger.error(f"Geen data na voorbereiding voor {pair} ({timeframe}), patroon {pattern_type}. Hopper over."); continue
                    target_label_column = f"{pattern_type}_label"
                    await self.train_ai_models(training_data=processed_df, symbol=pair, timeframe=timeframe, pattern_type=pattern_type, target_label_column=target_label_column)
                    # Time of day effectiveness can be analyzed on any of the processed_df, maybe just once after all loops?
                    # For now, keeping it here per pattern/pair/timeframe as it was.
                    await self.analyze_time_of_day_effectiveness(processed_df, strategy_id)

        logger.info(f"--- Pre-training en model training voltooid voor strategie {strategy_id} ---")

        # --- Start Backtesting Phase ---
        perform_backtesting = self.params_manager.get_param('perform_backtesting', False)
        if perform_backtesting and self.backtester:
            logger.info(f"--- Starten Backtesting Fase voor strategie: {strategy_id} ---")
            sequence_length_cnn = self.params_manager.get_param('sequence_length_cnn', 30)
            current_arch_key = self.params_manager.get_param('current_cnn_architecture_key', "default_simple")

            for pattern_type_bt in patterns_to_train_list: # Use the same list of patterns
                logger.info(f"--- Backtesting voor patroontype: {pattern_type_bt} ---")
                for pair_bt in pairs_to_fetch: # Use the same list of pairs
                    for timeframe_bt in timeframes_to_fetch: # Use the same list of timeframes
                        logger.info(f"--- Backtesten {pair_bt} ({timeframe_bt}) voor patroon: {pattern_type_bt}, architectuur: {current_arch_key} ---")
                        await self.backtester.run_backtest(
                            symbol=pair_bt,
                            timeframe=timeframe_bt,
                            pattern_type=pattern_type_bt,
                            architecture_key=current_arch_key,
                            sequence_length=sequence_length_cnn
                        )
            logger.info(f"--- Backtesting Fase voltooid voor strategie {strategy_id} ---")
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

        logger.info(f"--- STARTING INTEGRATION TEST FOR PRETRAINER (CV & BACKTESTING) ---")
        logger.info(f"Test configuration: Pairs={test_pairs_for_test}, Timeframe={test_timeframe}, Patterns={test_patterns_to_train}")

        # --- Artifact Clearing ---
        for pair_name in test_pairs_for_test:
            pair_sani = pair_name.replace('/', '_')
            # Clear raw data cache for this specific test pair/timeframe
            raw_data_cache_dir = Path(os.path.dirname(os.path.abspath(__file__))) / '..' / 'data' / 'raw_historical_data' / pair_sani / test_timeframe
            if raw_data_cache_dir.exists(): shutil.rmtree(raw_data_cache_dir)
            # Clear model output dir for this specific test pair/timeframe
            model_output_dir = Path(MODELS_DIR) / pair_sani / test_timeframe
            if model_output_dir.exists(): shutil.rmtree(model_output_dir)

        # Clear pretrain log file
        if os.path.exists(PRETRAIN_LOG_FILE): os.remove(PRETRAIN_LOG_FILE)

        # Clear backtest results file
        backtest_results_path = Path(MEMORY_DIR) / 'backtest_results.json'
        if backtest_results_path.exists():
            os.remove(backtest_results_path)
            logger.info(f"Verwijderde bestaand backtest resultatenbestand: {backtest_results_path}")
        logger.info(f"--- Test Setup: Artifacts Cleared (Raw Data Cache, Models, Pretrain Log, Backtest Log) ---")

        # --- ParamsManager Setup for Test ---
        params_manager_instance = ParamsManager()
        await params_manager_instance.set_param("data_fetch_start_date_str", "2023-11-01") # Adjusted for sufficient data
        await params_manager_instance.set_param("data_fetch_end_date_str", "2023-12-20")
        await params_manager_instance.set_param("data_fetch_pairs", test_pairs_for_test)
        await params_manager_instance.set_param("data_fetch_timeframes", [test_timeframe])
        await params_manager_instance.set_param("patterns_to_train", test_patterns_to_train)
        await params_manager_instance.set_param('sequence_length_cnn', 10)
        await params_manager_instance.set_param('num_epochs_cnn', 1)
        await params_manager_instance.set_param('batch_size_cnn', 8)

        # CV parameters
        await params_manager_instance.set_param('perform_cross_validation', True)
        await params_manager_instance.set_param('cv_num_splits', 2)

        # Backtesting parameters
        await params_manager_instance.set_param('perform_backtesting', True)
        await params_manager_instance.set_param('backtest_start_date_str', "2023-12-10") # Start backtest after training data
        await params_manager_instance.set_param('backtest_entry_threshold', 0.60)
        await params_manager_instance.set_param('backtest_take_profit_pct', 0.02)
        await params_manager_instance.set_param('backtest_stop_loss_pct', 0.01)
        await params_manager_instance.set_param('backtest_hold_duration_candles', 5)
        await params_manager_instance.set_param('backtest_initial_capital', 1000.0)
        await params_manager_instance.set_param('backtest_stake_pct_capital', 0.1)

        # Labeling config
        current_configs = params_manager_instance.get_param('pattern_labeling_configs')
        if not current_configs: current_configs = {}
        current_configs.setdefault("bullFlag", {}).update({ "future_N_candles": 5, "profit_threshold_pct": 0.005, "loss_threshold_pct": -0.003 })
        await params_manager_instance.set_param('pattern_labeling_configs', current_configs)

        # CNN Architecture
        await params_manager_instance.set_param('current_cnn_architecture_key', "default_simple")
        logger.info(f"TEST SETUP: ParamsManager ingesteld voor CV en Backtesting.")

        # --- PreTrainer Execution ---
        pre_trainer = PreTrainer(params_manager=params_manager_instance)
        test_strategy_id = "TestStrategy_CV_BT"

        print(f"\n--- EXECUTING PreTrainer Pipeline (CV & Backtesting enabled) ---")
        await pre_trainer.run_pretraining_pipeline(strategy_id=test_strategy_id, params_manager=params_manager_instance)

        # --- Validation of Outputs ---
        print(f"\n--- VALIDATION OF OUTPUTS (CV & Backtest Run) ---")
        arch_key_used = params_manager_instance.get_param('current_cnn_architecture_key')

        # Validate Model and Scaler files
        for pair_name in test_pairs_for_test:
            pair_sani = pair_name.replace('/', '_')
            for pattern in test_patterns_to_train:
                model_dir = Path(MODELS_DIR) / pair_sani / test_timeframe
                expected_model_file = model_dir / f'cnn_model_{pattern}_{arch_key_used}.pth'
                expected_scaler_file = model_dir / f'scaler_params_{pattern}_{arch_key_used}.json'

                if expected_model_file.exists(): print(f"SUCCESS: Model file for {pair_sani}/{pattern} (arch: {arch_key_used}) created.")
                else: print(f"FAILURE: Model file for {pair_sani}/{pattern} (arch: {arch_key_used}) NOT created: {expected_model_file}")

                if expected_scaler_file.exists(): print(f"SUCCESS: Scaler file for {pair_sani}/{pattern} (arch: {arch_key_used}) created.")
                else: print(f"FAILURE: Scaler file for {pair_sani}/{pattern} (arch: {arch_key_used}) NOT created: {expected_scaler_file}")

        # Validate Pretrain Log for CV results
        if os.path.exists(PRETRAIN_LOG_FILE):
            with open(PRETRAIN_LOG_FILE, 'r') as f: log_entries = json.load(f)
            if log_entries and isinstance(log_entries, list) and len(log_entries) > 0:
                last_log_entry = log_entries[-1] # Assuming last entry corresponds to the test run
                if 'cross_validation_results' in last_log_entry:
                    print(f"SUCCESS: 'cross_validation_results' found in the last pretrain log entry: {last_log_entry['cross_validation_results']}")
                    assert last_log_entry['cross_validation_results']['num_splits'] == params_manager_instance.get_param('cv_num_splits')
                else:
                    print(f"FAILURE: 'cross_validation_results' not found in the last pretrain log entry.")
            else: print(f"FAILURE: Pretrain Log file {PRETRAIN_LOG_FILE} is empty or not a list.")
        else: print(f"FAILURE: Pretrain Log file {PRETRAIN_LOG_FILE} not found.")

        # Validate Backtest Results File
        if backtest_results_path.exists():
            with open(backtest_results_path, 'r') as f:
                try:
                    backtest_log_entries = json.load(f)
                    if backtest_log_entries and isinstance(backtest_log_entries, list) and len(backtest_log_entries) > 0:
                        print(f"SUCCESS: Backtest results file created and contains {len(backtest_log_entries)} entries.")
                        first_bt_entry = backtest_log_entries[0] # Check the first entry

                        # Basic assertions from previous step
                        assert first_bt_entry['symbol'] == test_pairs_for_test[0]
                        assert first_bt_entry['timeframe'] == test_timeframe
                        assert first_bt_entry['pattern_type'] == test_patterns_to_train[0]
                        assert first_bt_entry['architecture_key'] == arch_key_used
                        print(f"First backtest entry - Basic info: Symbol={first_bt_entry['symbol']}, Return={first_bt_entry['metrics']['total_return_pct']:.2f}%")

                        # Verify presence of new financial metrics
                        expected_metrics = [
                            'initial_capital', 'final_capital', 'total_return_pct', 'total_pnl_from_trades',
                            'num_trades', 'num_wins', 'num_losses', 'win_rate_pct', 'loss_rate_pct',
                            'average_pnl_per_trade', 'avg_profit_per_winning_trade', 'avg_loss_per_losing_trade',
                            'profit_factor', 'sharpe_ratio_annualized', 'max_drawdown_pct'
                        ]
                        missing_metrics = [m for m in expected_metrics if m not in first_bt_entry['metrics']]
                        if not missing_metrics:
                            print(f"SUCCESS: All expected new financial metrics are present in the first backtest entry.")
                            print(f"  Sharpe Ratio: {first_bt_entry['metrics'].get('sharpe_ratio_annualized')}")
                            print(f"  Max Drawdown: {first_bt_entry['metrics'].get('max_drawdown_pct')}")
                            print(f"  Profit Factor: {first_bt_entry['metrics'].get('profit_factor')}")
                        else:
                            print(f"FAILURE: Missing financial metrics in backtest entry: {missing_metrics}")
                            assert not missing_metrics, f"Missing metrics: {missing_metrics}"

                        # Verify params_used structure
                        if 'params_used' in first_bt_entry:
                            print(f"SUCCESS: 'params_used' key found in backtest entry.")
                        else:
                            print(f"FAILURE: 'params_used' key NOT found in backtest entry.")
                            assert 'params_used' in first_bt_entry, "'params_used' key missing"

                    else:
                        print(f"FAILURE: Backtest results file {backtest_results_path} is empty or has an unexpected format (not a non-empty list).")
                except json.JSONDecodeError:
                    print(f"FAILURE: Could not decode JSON from backtest results file {backtest_results_path}.")
        else:
            print(f"FAILURE: Backtest results file {backtest_results_path} not found.")

        # --- Optional: Test SimpleCNN Custom Architecture Instantiation ---
        print("\n--- Testing SimpleCNN Custom Architecture Instantiation ---")
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
