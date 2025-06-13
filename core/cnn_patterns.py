# core/cnn_patterns.py
import logging
import os
from sklearn.preprocessing import MinMaxScaler
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import talib # Voor candlestick patronen die TA-Lib biedt

# PyTorch Imports
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
            block_layers.append(nn.Conv1d(in_channels=current_channels,
                                          out_channels=filters_per_layer[i],
                                          kernel_size=kernel_sizes_per_layer[i],
                                          stride=strides_per_layer[i],
                                          padding=padding_per_layer[i]))
            if use_batch_norm:
                block_layers.append(nn.BatchNorm1d(filters_per_layer[i]))
            block_layers.append(nn.ReLU())
            if pooling_types_per_layer[i] and pooling_types_per_layer[i].lower() != 'none':
                pool_kernel = pooling_kernel_sizes_per_layer[i]
                pool_stride = pooling_strides_per_layer[i]
                if pooling_types_per_layer[i].lower() == 'max':
                    block_layers.append(nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride))
                elif pooling_types_per_layer[i].lower() == 'avg':
                    block_layers.append(nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride))

            self.conv_blocks.append(nn.Sequential(*block_layers))
            current_channels = filters_per_layer[i]

        with torch.no_grad():
            dummy_x = torch.randn(1, self.input_channels, self.sequence_length)
            for block in self.conv_blocks:
                dummy_x = block(dummy_x)
            fc_input_features = dummy_x.view(dummy_x.size(0), -1).shape[1]

        self.dropout_layer = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.fc = nn.Linear(fc_input_features, num_classes)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        if self.dropout_layer:
            x = self.dropout_layer(x)
        x = self.fc(x)
        return x

class CNNPatterns:
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'models')

    def __init__(self):
        from core.params_manager import ParamsManager
        self.params_manager = ParamsManager()
        self.models: Dict[str, SimpleCNN] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}

        self.pattern_model_meta_info = {
            'bullFlag': {'input_channels': 12, 'num_classes': 2},
            'bearishEngulfing': {'input_channels': 12, 'num_classes': 2}
        }
        logger.info("CNNPatterns geïnitialiseerd. Modellen worden on-demand geladen via predict_pattern_score.")

    def _load_cnn_models_and_scalers(self, symbol: str, timeframe: str, arch_key_to_load: Optional[str] = None):
        symbol_underscore = symbol.replace('/', '_')

        if arch_key_to_load is None: # If no specific arch key is passed, use current from PM
            arch_key_to_load = self.params_manager.get_param('current_cnn_architecture_key', "default_simple")

        logger.info(f"Attempting to load models/scalers for {symbol_underscore}/{timeframe} using architecture '{arch_key_to_load}'.")

        all_arch_configs = self.params_manager.get_param('cnn_architecture_configs')
        arch_params = {} # Default to empty dict -> SimpleCNN uses its internal defaults

        if all_arch_configs and arch_key_to_load in all_arch_configs:
            arch_params = all_arch_configs[arch_key_to_load]
        elif arch_key_to_load != "default_simple":
            logger.warning(f"Architecture '{arch_key_to_load}' not found in ParamsManager. Cannot load specific models for {symbol_underscore}/{timeframe}/{arch_key_to_load}.")
            return
        else:
            logger.warning(f"Default architecture 'default_simple' (or requested '{arch_key_to_load}') not found in configs. SimpleCNN will use its internal defaults for instantiation if needed.")


        for pattern_name, meta_info in self.pattern_model_meta_info.items():
            model_dir = os.path.join(CNNPatterns.MODELS_DIR, symbol_underscore, timeframe)

            model_filename = f"cnn_model_{pattern_name}_{arch_key_to_load}.pth"
            scaler_filename = f"scaler_params_{pattern_name}_{arch_key_to_load}.json"

            model_path = os.path.join(model_dir, model_filename)
            scaler_path = os.path.join(model_dir, scaler_filename)

            model_key = f"{symbol_underscore}_{timeframe}_{pattern_name}_{arch_key_to_load}"

            if model_key in self.models:
                continue

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'r', encoding='utf-8') as f:
                        scaler_params_json = json.load(f)

                    sequence_length = scaler_params_json.get('sequence_length', 30)

                    num_input_features = meta_info['input_channels']
                    if 'feature_names_in_' in scaler_params_json and isinstance(scaler_params_json['feature_names_in_'], list):
                        num_input_features = len(scaler_params_json['feature_names_in_'])
                    elif 'min_' in scaler_params_json and isinstance(scaler_params_json['min_'], list):
                        num_input_features = len(scaler_params_json['min_'])

                    model = SimpleCNN(
                        input_channels=num_input_features,
                        num_classes=meta_info['num_classes'],
                        sequence_length=sequence_length,
                        **arch_params
                    )
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    model.eval()
                    self.models[model_key] = model
                    logger.info(f"CNN model '{model_key}' succesvol geladen van {model_path}.")

                    scaler = MinMaxScaler()
                    scaler.min_ = np.array(scaler_params_json['min_'])
                    scaler.scale_ = np.array(scaler_params_json['scale_'])

                    if 'feature_names_in_' in scaler_params_json and isinstance(scaler_params_json['feature_names_in_'], list):
                        scaler.feature_names_in_ = np.array(scaler_params_json['feature_names_in_'])
                        scaler.n_features_in_ = len(scaler_params_json['feature_names_in_'])
                    else:
                        scaler.n_features_in_ = num_input_features
                        logger.warning(f"'feature_names_in_' niet gevonden in scaler_params voor '{model_key}'. n_features_in_ afgeleid als {num_input_features}.")

                    self.scalers[model_key] = scaler
                    logger.info(f"Scaler parameters voor '{model_key}' succesvol geladen van {scaler_path}.")

                except Exception as e:
                    logger.error(f"Fout bij laden CNN model of scaler voor '{model_key}': {e}.", exc_info=True)
            else:
                logger.debug(f"CNN model of scaler niet gevonden voor '{model_key}'. Verwachte paden: Model='{model_path}', Scaler='{scaler_path}'.")

    def _dataframe_to_cnn_input(self, dataframe: pd.DataFrame, model_key: str, sequence_length: int) -> Optional[torch.Tensor]:
        scaler = self.scalers.get(model_key)
        if scaler is None:
            logger.warning(f"Scaler niet geladen voor model_key '{model_key}'. Kan CNN input niet correct normaliseren.")
            return None

        if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
            features_cols = list(scaler.feature_names_in_)
        else:
            features_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
            logger.warning(f"Scaler for '{model_key}' has no 'feature_names_in_'. Using default 12-feature list.")

        temp_df = dataframe.copy()
        for col in features_cols:
            if col not in temp_df.columns:
                logger.debug(f"Kolom '{col}' niet gevonden in dataframe voor model_key '{model_key}'. Wordt toegevoegd met NaN.")
                temp_df[col] = np.nan

        if len(temp_df) < sequence_length:
            logger.debug(f"Niet genoeg candles ({len(temp_df)}) voor CNN input (nodig: {sequence_length}) voor model_key '{model_key}'.")
            return None

        input_data_pd = temp_df[features_cols].tail(sequence_length)

        if input_data_pd.isnull().values.any():
            logger.warning(f"NaN waarden gevonden in input data voor CNN voor model_key '{model_key}'.")
            return None

        input_data_np = input_data_pd.values

        try:
            X_scaled = scaler.transform(input_data_np)
        except Exception as e:
            logger.error(f"Fout bij toepassen van scaler voor model_key '{model_key}': {e}")
            logger.error(f"Scaler details: min: {scaler.min_}, scale: {scaler.scale_}, n_features_in: {getattr(scaler, 'n_features_in_', 'N/A')}")
            logger.error(f"Input data shape: {input_data_np.shape}, expected features: {features_cols}")
            return None

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
        return X_tensor

    async def predict_pattern_score(self, dataframe: pd.DataFrame, symbol: str, timeframe: str, pattern_name: str) -> float:
        """
        Uses the loaded CNN model to predict a numerical score (probability)
        for a specific pattern, symbol, and timeframe.
        It will attempt to load the model if not already available for the current architecture.
        """
        current_arch_key = self.params_manager.get_param('current_cnn_architecture_key', "default_simple")
        model_key = f"{symbol.replace('/', '_')}_{timeframe}_{pattern_name}_{current_arch_key}"

        model = self.models.get(model_key)
        scaler = self.scalers.get(model_key)

        if model is None or scaler is None:
            logger.info(f"Model of scaler voor '{model_key}' niet direct gevonden. Poging tot laden...")
            self._load_cnn_models_and_scalers(symbol, timeframe, current_arch_key)
            model = self.models.get(model_key)
            scaler = self.scalers.get(model_key)
            if model is None or scaler is None:
                logger.warning(f"CNN model of scaler voor '{model_key}' is niet geladen na poging. Kan geen score voorspellen.")
                return 0.0

        model_sequence_length = model.sequence_length

        cnn_input_tensor = self._dataframe_to_cnn_input(dataframe, model_key, sequence_length=model_sequence_length)

        if cnn_input_tensor is None:
            logger.debug(f"Kon geen CNN input tensor genereren voor model '{model_key}'.")
            return 0.0

        try:
            with torch.no_grad():
                output = model(cnn_input_tensor)
                probabilities = torch.softmax(output, dim=1)
                score = probabilities[0, 1].item()
                logger.debug(f"CNN voorspelde score voor '{model_key}': {score:.4f}")
                return score
        except Exception as e:
            logger.error(f"Fout bij CNN-voorspelling voor '{model_key}': {e}")
            return 0.0

    def get_all_detectable_pattern_keys(self) -> List[str]:
        """
        Returns a list of all pattern keys that the CNN models can detect.
        These keys correspond to the names of the trained models.
        """
        return list(self.pattern_model_meta_info.keys())

    # --- Helperfuncties voor dataverwerking (uitbreiding van Freqtrade DF naar 'candles' dicts) ---
    def _dataframe_to_candles(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Converteert een Freqtrade DataFrame naar een lijst van 'candle'-dictionaries.
        Dit is nodig omdat sommige JS detectiefuncties werkten met dictionaries.
        Freqtrade OHLCV is: timestamp, open, high, low, close, volume.
        """
        df_copy = dataframe.copy()
        if 'date' not in df_copy.columns:
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index()
                if 'index' in df_copy.columns and 'date' not in df_copy.columns:
                     df_copy.rename(columns={'index': 'date'}, inplace=True)
                elif df_copy.index.name == 'date' and 'date' not in df_copy.columns:
                    df_copy.rename(columns={df_copy.index.name: 'date'}, inplace=True)
            else:
                logger.error("DataFrame heeft geen 'date' kolom of DatetimeIndex.")
                return []
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            try: df_copy['date'] = pd.to_datetime(df_copy['date'])
            except Exception as e: logger.error(f"Kan 'date' kolom niet converteren naar datetime: {e}"); return []
        candles = []
        for _, row in df_copy.iterrows():
            candle = {
                'time': row['date'].timestamp() * 1000,
                'open': float(row['open']), 'high': float(row['high']),
                'low': float(row['low']), 'close': float(row['close']),
                'volume': float(row['volume']),
                'quoteVolume': float(row.get('quote_volume', 0.0)),
                'rsi': float(row.get('rsi', np.nan)), 'macd': float(row.get('macd', np.nan)),
                'macdsignal': float(row.get('macdsignal', np.nan)), 'macdhist': float(row.get('macdhist', np.nan)),
                'bb_upperband': float(row.get('bb_upperband', np.nan)),
                'bb_middleband': float(row.get('bb_middleband', np.nan)),
                'bb_lowerband': float(row.get('bb_lowerband', np.nan)),
            }
            candles.append(candle)
        return candles

    def _mean(self, values: List[float]) -> float:
        if not values: return 0.0
        valid_values = [v for v in values if not np.isnan(v)]
        if not valid_values: return 0.0
        return sum(valid_values) / len(valid_values)

    # --- Candlestick Patroon Detectie Functies ---
    def detect_bull_flag(self, candles: List[Dict[str, Any]]) -> bool:
        if len(candles) < 10: return False
        pole_candles = candles[-10:-5]
        flag_candles = candles[-5:]
        if not all(c['close'] > c['open'] for c in pole_candles if c is not None): return False
        flag_closes = np.array([c['close'] for c in flag_candles if c is not None])
        if len(flag_closes) < 2 : return False
        if flag_closes[-1] < flag_closes[0] and (flag_closes[0] - flag_closes[-1]) / (flag_closes[0] + 1e-9) < 0.03:
            return True
        return False

    def detect_double_top(self, candles: List[Dict[str, Any]]) -> bool:
        if len(candles) < 20: return False
        highs = np.array([c['high'] for c in candles[-20:]])
        if len(highs) < 20 : return False
        first_half_highs = highs[:len(highs)//2]; second_half_highs = highs[len(highs)//2:]
        if not first_half_highs.size or not second_half_highs.size: return False
        peak1_idx = np.argmax(first_half_highs)
        peak2_idx = np.argmax(second_half_highs) + len(highs)//2
        if abs(highs[peak1_idx] - highs[peak2_idx]) / (highs[peak1_idx] + 1e-9) < 0.01:
            if peak1_idx < peak2_idx and min(highs[peak1_idx:peak2_idx]) < highs[peak1_idx] * 0.95:
                return True
        return False

    def detect_head_and_shoulders(self, candles: List[Dict[str, Any]]) -> bool:
        logger.warning(f"{self.__class__.__name__}.detect_head_and_shoulders is a placeholder and not fully implemented.")
        if len(candles) < 30: return False
        return False

    def detect_breakout(self, candles: List[Dict[str, Any]], resistance_level: Optional[float] = None) -> bool:
        if len(candles) < 2: return False; last_candle = candles[-1]
        if resistance_level is None:
            prev_highs = [c['high'] for c in candles[-10:-1] if c is not None]
            if not prev_highs: return False
            resistance_level = max(prev_highs)
        current_volume = last_candle.get('volume', 0)
        avg_volume_recent = self._mean([c.get('volume', 0) for c in candles[-20:-1] if c is not None])
        return last_candle['close'] > resistance_level and current_volume > avg_volume_recent * 1.5

    def detect_ema_cross(self, candles: List[Dict[str, Any]], short_period: int = 12, long_period: int = 26) -> str or bool:
        if len(candles) < max(short_period, long_period): return False
        closes = np.array([c['close'] for c in candles if c is not None])
        if len(closes) < max(short_period, long_period): return False
        ema_short = pd.Series(closes).ewm(span=short_period, adjust=False).mean().iloc[-1]
        ema_long = pd.Series(closes).ewm(span=long_period, adjust=False).mean().iloc[-1]
        prev_closes = np.array([c['close'] for c in candles[:-1] if c is not None])
        if len(prev_closes) < max(short_period, long_period): return False
        prev_ema_short = pd.Series(prev_closes).ewm(span=short_period, adjust=False).mean().iloc[-1]
        prev_ema_long = pd.Series(prev_closes).ewm(span=long_period, adjust=False).mean().iloc[-1]
        if ema_short > ema_long and prev_ema_short <= prev_ema_long: return "bullishCross"
        if ema_short < ema_long and prev_ema_short >= prev_ema_long: return "bearishCross"
        return False

    def detect_cup_and_handle(self, candles: List[Dict[str, Any]]) -> bool:
        logger.warning(f"{self.__class__.__name__}.detect_cup_and_handle is a placeholder and not fully implemented.")
        if len(candles) < 50: return False; return False
    def detect_wedge_patterns(self, candles: List[Dict[str, Any]]) -> bool:
        logger.warning(f"{self.__class__.__name__}.detect_wedge_patterns is a placeholder and not fully implemented.")
        if len(candles) < 20: return False; return False
    def detect_triple_top_bottom(self, candles: List[Dict[str, Any]]) -> bool:
        logger.warning(f"{self.__class__.__name__}.detect_triple_top_bottom is a placeholder and not fully implemented.")
        if len(candles) < 30: return False; return False

    def detect_candlestick_patterns(self, candles_df: pd.DataFrame) -> Dict[str, bool]:
        patterns = {};
        if len(candles_df) < 1: return {}
        open_, high_, low_, close_ = candles_df['open'].astype(float), candles_df['high'].astype(float), candles_df['low'].astype(float), candles_df['close'].astype(float)
        try: patterns['CDLDOJI'] = bool(talib.CDLDOJI(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLDOJI.lookback else False
        except Exception: patterns['CDLDOJI'] = False
        try: patterns['CDLENGULFING'] = bool(talib.CDLENGULFING(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLENGULFING.lookback else False # Example
        except Exception: patterns['CDLENGULFING'] = False
        return {k: v for k, v in patterns.items() if v is True}

    def _detect_engulfing(self, candles: List[Dict[str, Any]], direction: str = "any") -> str or bool: # Added direction
        if len(candles) < 2: return False
        prev, curr = candles[-2], candles[-1]
        if prev is None or curr is None: return False
        is_bullish_engulfing = prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['open'] and curr['open'] < prev['close']
        is_bearish_engulfing = prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] > prev['close'] and curr['close'] < prev['open']
        if direction == "bullish" and is_bullish_engulfing: return "bullishEngulfing"
        if direction == "bearish" and is_bearish_engulfing: return "bearishEngulfing"
        if direction == "any":
            if is_bullish_engulfing: return "bullishEngulfing"
            if is_bearish_engulfing: return "bearishEngulfing"
        return False

    # ... (rest of the pattern detection methods from the original file, simplified for brevity if they were very long) ...
    def _detect_morning_evening_star(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 3: return False; a,b,c = candles[-3:]; body = lambda x: abs(x['close']-x['open'])
        if None in [a,b,c] or body(a) < 1e-9: return False
        bullish = a['close'] < a['open'] and body(b) < (body(a)/2) and c['close'] > c['open'] and c['close'] > ((a['open']+a['close'])/2)
        bearish = a['close'] > a['open'] and body(b) < (body(a)/2) and c['close'] < c['open'] and c['close'] < ((a['open']+a['close'])/2)
        return "morningStar" if bullish else ("eveningStar" if bearish else False)

    def _detect_three_white_soldiers(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 3: return False; a,b,c = candles[-3:]; body_perc = lambda x: abs(x['close']-x['open'])/(x['high']-x['low']+1e-9)
        if None in [a,b,c]: return False
        if all(x['close'] > x['open'] for x in [a,b,c]) and \
           b['open'] > a['open'] and c['open'] > b['open'] and \
           b['close'] > a['close'] and c['close'] > b['close'] and \
           all(body_perc(x) > 0.6 for x in [a,b,c]): return "threeWhiteSoldiers"
        return False

    def _detect_three_black_crows(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 3: return False; a,b,c = candles[-3:]; body_perc = lambda x: abs(x['close']-x['open'])/(x['high']-x['low']+1e-9)
        if None in [a,b,c]: return False
        if all(x['close'] < x['open'] for x in [a,b,c]) and \
           b['open'] < a['open'] and c['open'] < b['open'] and \
           b['close'] < a['close'] and c['close'] < b['close'] and \
           all(body_perc(x) > 0.6 for x in [a,b,c]): return "threeBlackCrows"
        return False

    def determine_trend(self, candles: List[Dict[str, Any]], period: int = 20) -> str:
        """
        Determines the trend based on linear regression of closing prices.
        """
        if len(candles) < period:
            return "undetermined"

        # Extract closing prices from the last 'period' candles
        # Ensure 'close' key exists and value is not None
        closes = [c['close'] for c in candles[-period:] if c and 'close' in c and c['close'] is not None]

        if len(closes) < period: # Need enough data points for a reliable trend
            return "undetermined"

        # Calculate linear regression
        x = np.arange(len(closes))
        y = np.array(closes)

        # Fit linear regression: y = slope * x + intercept
        # Using np.polyfit for simplicity
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except (np.linalg.LinAlgError, TypeError): # Catch potential errors during polyfit
            logger.warning(f"Could not calculate polyfit for trend determination. Closes: {closes}")
            return "undetermined"

        mean_price = np.mean(y)
        if mean_price == 0: # Avoid division by zero
            return "undetermined"

        # Define a threshold for "sideways"
        # e.g., if abs(slope / mean_price) < 0.0005 (0.05% change per candle relative to mean price)
        # This threshold might need tuning based on timeframe and asset volatility
        sideways_threshold = 0.0005 * period # Adjust threshold based on period length

        normalized_slope = slope / mean_price

        if abs(normalized_slope) < sideways_threshold:
            return "sideways"
        elif slope > 0:
            return "uptrend"
        else:
            return "downtrend"

    def detect_volume_spike(self, candles: List[Dict[str, Any]], period: int = 20, spike_factor: float = 2.0) -> bool:
        """
        Detects a volume spike in the most recent candle compared to the average of previous candles.
        """
        if len(candles) < period:
            return False # Not enough data to compare

        # Ensure 'volume' key exists and value is not None for all relevant candles
        volumes = [c['volume'] for c in candles[-(period):] if c and 'volume' in c and c['volume'] is not None]

        if len(volumes) < period: # Need full period of valid volumes
            return False

        latest_volume = volumes[-1]
        preceding_volumes = volumes[:-1]

        if not preceding_volumes: # Should not happen if len(volumes) == period and period > 1
            return False

        average_volume_preceding = np.mean(preceding_volumes)

        if average_volume_preceding == 0:
            return False # Avoid division by zero or if all previous volumes were zero

        if latest_volume > (average_volume_preceding * spike_factor):
            return True

        return False

    async def detect_patterns_multi_timeframe(self, candles_by_timeframe: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        all_patterns = {"zoomPatterns": {}, "contextPatterns": {}, "patterns": {}, "cnn_predictions": {}, "context": {"trend": "unknown", "volume_spike": False}}
        TIME_FRAME_CONFIG = {"zoom": ['1m', '5m', '15m'], "context": ['1h', '4h', '1d']} # Simplified for brevity

        for tf_type, timeframes in TIME_FRAME_CONFIG.items():
            for tf in timeframes:
                if tf in candles_by_timeframe and not candles_by_timeframe[tf].empty:
                    candles_list = self._dataframe_to_candles(candles_by_timeframe[tf])
                    if not candles_list: continue

                    # CNN Predictions
                    for pattern_name_cnn in self.pattern_model_meta_info.keys():
                        score = await self.predict_pattern_score(candles_by_timeframe[tf], symbol, tf, pattern_name_cnn)
                        if score > 0.0: # Log if score is meaningful
                            all_patterns["cnn_predictions"][f"{tf}_{pattern_name_cnn}_score"] = score

                    # Simplified rule-based for example
                    detected_tf_patterns = {}
                    if self.detect_bull_flag(candles_list): detected_tf_patterns["bullFlag"] = True
                    eng_res = self._detect_engulfing(candles_list, "any")
                    if eng_res: detected_tf_patterns[eng_res] = True # Stores "bullishEngulfing" or "bearishEngulfing"

                    talib_patterns = self.detect_candlestick_patterns(candles_by_timeframe[tf])
                    detected_tf_patterns.update(talib_patterns)

                    if detected_tf_patterns:
                        all_patterns[tf_type][tf] = {k:v for k,v in detected_tf_patterns.items() if v}

        # Consolidate patterns (simplified)
        for tf_type_key in ["zoomPatterns", "contextPatterns"]:
            for patterns_on_tf in all_patterns[tf_type_key].values():
                for pattern_name, status in patterns_on_tf.items():
                    if status: all_patterns["patterns"][pattern_name] = status

        # Determine context (simplified)
        main_context_tf = next((tf for tf in ['1h', '4h', '1d'] if tf in candles_by_timeframe and not candles_by_timeframe[tf].empty), None)
        if main_context_tf:
            context_candles = self._dataframe_to_candles(candles_by_timeframe[main_context_tf])
            if context_candles:
                all_patterns["context"]["trend"] = self.determine_trend(context_candles)
                all_patterns["context"]["volume_spike"] = self.detect_volume_spike(context_candles)

        logger.debug(f"Patroondetectie resultaat voor {symbol}: {json.dumps(all_patterns, indent=2, default=str)}")
        return all_patterns
