# core/cnn_patterns.py
import logging
import os
from sklearn.preprocessing import MinMaxScaler
import json
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
# Future Improvement: The current training parameters (e.g., epochs, learning rate, batch_size)
# and model architecture are foundational. Systematic hyperparameter tuning and experimentation
# with different CNN architectures should be conducted to optimize performance for specific patterns.
    def __init__(self, input_channels, num_classes, sequence_length=30): # Added sequence_length
        super(SimpleCNN, self).__init__()
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=self.input_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # Added stride
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # Added stride

        # Dynamically calculate the number of features for the fully connected layer
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, sequence_length)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            fc_input_features = x.numel() # Calculate the total number of elements

        self.fc = nn.Linear(fc_input_features, self.num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # Flatten the output from conv/pool layers
        # The view shape is (batch_size, fc_input_features)
        x = x.view(x.size(0), -1) # Dynamically flatten
        x = self.fc(x)
        return x

class CNNPatterns:
# Future Improvement: For optimal ML performance, consider enhancing training label quality.
# Current algorithmic labeling is a good starting point, but more refined labels,
# potentially annotated with domain expertise, could significantly improve model accuracy and robustness.
    """
    Detecteert visuele candlestick- en technische patronen in marktdata.
    Integreert nu een PyTorch CNN model voor patroonherkenning.
    """

    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'models')

    def __init__(self):
        self.models: Dict[str, SimpleCNN] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.pattern_configs = {
            'bullFlag': {'input_channels': 9, 'num_classes': 2, 'model_path': os.path.join(CNNPatterns.MODELS_DIR, 'cnn_model_bullFlag.pth'), 'scaler_path': os.path.join(CNNPatterns.MODELS_DIR, 'scaler_params_bullFlag.json')},
            'bearishEngulfing': {'input_channels': 9, 'num_classes': 2, 'model_path': os.path.join(CNNPatterns.MODELS_DIR, 'cnn_model_bearishEngulfing.pth'), 'scaler_path': os.path.join(CNNPatterns.MODELS_DIR, 'scaler_params_bearishEngulfing.json')}
        }
        self._load_cnn_models_and_scalers()
        logger.info("CNNPatterns geïnitialiseerd.")

    def _load_cnn_models_and_scalers(self):
        """Laadt alle getrainde CNN-modellen en hun bijbehorende scalers."""
        for pattern_name, config in self.pattern_configs.items():
            model_path = config['model_path']
            scaler_path = config['scaler_path']

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    # Laad scaler parameters en reconstrueer scaler
                    with open(scaler_path, 'r', encoding='utf-8') as f:
                        scaler_params = json.load(f)

                    sequence_length = scaler_params.get('sequence_length', 30) # Get sequence_length, default to 30

                    # Laad model
                    model = SimpleCNN(input_channels=config['input_channels'], num_classes=config['num_classes'], sequence_length=sequence_length)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    self.models[pattern_name] = model
                    logger.info(f"CNN model voor '{pattern_name}' succesvol geladen van {model_path} met sequence_length={sequence_length}.")

                    scaler = MinMaxScaler() # Requires MinMaxScaler import
                    scaler.min_ = np.array(scaler_params['min_'])
                    scaler.scale_ = np.array(scaler_params['scale_'])

                    # Set n_features_in_ and feature_names_in_
                    # Prefer 'feature_names_in_' if available, else 'features_cols'
                    if 'feature_names_in_' in scaler_params and isinstance(scaler_params['feature_names_in_'], list):
                        scaler.n_features_in_ = len(scaler_params['feature_names_in_'])
                        scaler.feature_names_in_ = np.array(scaler_params['feature_names_in_'])
                    elif 'features_cols' in scaler_params and isinstance(scaler_params['features_cols'], list): # Fallback for older params format
                        scaler.n_features_in_ = len(scaler_params['features_cols'])
                        scaler.feature_names_in_ = np.array(scaler_params['features_cols'])
                    else:
                        # Attempt to infer n_features_in_ from min_ or scale_ if names are not present
                        # This is a fallback and assumes min_ or scale_ are correctly populated lists/arrays
                        if scaler_params.get('min_') is not None:
                             scaler.n_features_in_ = len(scaler_params['min_'])
                        elif scaler_params.get('scale_') is not None:
                             scaler.n_features_in_ = len(scaler_params['scale_'])
                        logger.warning(f"Feature names ('feature_names_in_' or 'features_cols') not found or not a list in scaler_params for '{pattern_name}'. n_features_in_ inferred as {getattr(scaler, 'n_features_in_', 'Unknown')}. This might lead to issues if feature order/names are critical.")

                    self.scalers[pattern_name] = scaler
                    logger.info(f"Scaler parameters voor '{pattern_name}' succesvol geladen van {scaler_path}.")

                except Exception as e:
                    logger.error(f"Fout bij laden CNN model of scaler voor '{pattern_name}': {e}. Dit model zal niet beschikbaar zijn.")
                    self.models[pattern_name] = None # type: ignore
                    self.scalers[pattern_name] = None # type: ignore
            else:
                logger.warning(f"CNN model of scaler niet gevonden voor '{pattern_name}' (verwacht: {model_path}, {scaler_path}). CNN-voorspellingen voor dit patroon zullen niet beschikbaar zijn.")
                self.models[pattern_name] = None # type: ignore
                self.scalers[pattern_name] = None # type: ignore

    def _dataframe_to_cnn_input(self, dataframe: pd.DataFrame, pattern_name: str, sequence_length: int = 30) -> Optional[torch.Tensor]:
        """
        Converteert een Pandas DataFrame naar een tensor formaat geschikt voor de CNN.
        Gebruikt de correcte, geladen scaler voor normalisatie.
        """
        scaler = self.scalers.get(pattern_name)
        if scaler is None:
            logger.warning(f"Scaler niet geladen voor patroon '{pattern_name}'. Kan CNN input niet correct normaliseren.")
            return None

        # Original features_cols from the issue.
        # Ensure these are consistent with what models were trained on.
        features_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist']

        # Check if all features_cols exist, if not, add them with NaNs
        # This is important if a dataframe for a particular timeframe might be missing some indicators
        temp_df = dataframe.copy() # Work on a copy
        for col in features_cols:
            if col not in temp_df.columns:
                logger.debug(f"Kolom '{col}' niet gevonden in dataframe voor patroon '{pattern_name}'. Wordt toegevoegd met NaN.")
                temp_df[col] = np.nan # Use np.nan for numerical columns

        if len(temp_df) < sequence_length:
            logger.debug(f"Niet genoeg candles ({len(temp_df)}) voor CNN input (nodig: {sequence_length}) voor patroon '{pattern_name}'.")
            return None

        # Select the relevant part of the dataframe and the features
        # Use .tail(sequence_length) to get the last N rows
        input_data_pd = temp_df[features_cols].tail(sequence_length)

        # Check for NaNs after selection and before scaling
        if input_data_pd.isnull().values.any():
            logger.warning(f"NaN waarden gevonden in input data voor CNN voor patroon '{pattern_name}' na selectie van features en sequence. Input data:
{input_data_pd}")
            # Optionally, decide how to handle NaNs: fill, drop, or return None
            # For now, returning None as scaling would fail or produce NaNs in tensor
            return None

        input_data_np = input_data_pd.values

        # Reshape for scaler: (samples * sequence_length, features) -> in this case (sequence_length, features)
        # The scaler expects 2D data where the second dimension is n_features.
        # input_data_np is already (sequence_length, n_features)

        try:
            # Use the loaded scaler to transform
            # Scaler expects data in shape (n_samples, n_features)
            # Here, n_samples is sequence_length
            X_scaled = scaler.transform(input_data_np)
        except Exception as e:
            logger.error(f"Fout bij toepassen van scaler voor patroon '{pattern_name}': {e}")
            logger.error(f"Scaler details: min: {scaler.min_}, scale: {scaler.scale_}")
            logger.error(f"Input data shape: {input_data_np.shape}, first few rows of input_data_np:
{input_data_np[:3]}")
            # Check if feature_names_in_ matches input_data_pd.columns
            if hasattr(scaler, 'feature_names_in_') and list(scaler.feature_names_in_) != features_cols:
                logger.error(f"Mismatch in scaler's feature_names_in_ {list(scaler.feature_names_in_)} and expected features_cols {features_cols}")
            return None

        # X_normalized is X_scaled, no reshape needed here as X_scaled is already (sequence_length, features)

        # Convert to PyTorch tensor and reshape to (1, channels, sequence_length)
        # channels = num_features
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
        return X_tensor

    async def predict_pattern_score(self, dataframe: pd.DataFrame, pattern_name: str) -> float:
        """
        Gebruikt het geladen CNN-model om een numerieke score (waarschijnlijkheid)
        te voorspellen voor een specifiek patroon op basis van de input dataframe.
        """
        model = self.models.get(pattern_name)
        if model is None:
            logger.debug(f"CNN model is niet geladen voor '{pattern_name}'. Kan geen score voorspellen.")
            return 0.0

        # Call the new _dataframe_to_cnn_input method
        # Use the sequence_length specific to the loaded model
        model_sequence_length = model.sequence_length
        cnn_input_tensor = self._dataframe_to_cnn_input(dataframe, pattern_name, sequence_length=model_sequence_length)

        if cnn_input_tensor is None:
            logger.debug(f"Kon geen CNN input tensor genereren voor patroon '{pattern_name}'.")
            return 0.0

        try:
            with torch.no_grad():
                output = model(cnn_input_tensor)
                # Assuming output is (batch_size, num_classes)
                # And positive class is at index 1
                probabilities = torch.softmax(output, dim=1)
                score = probabilities[0, 1].item() # Probability of the positive class (e.g., pattern detected)
                logger.debug(f"CNN voorspelde score voor '{pattern_name}': {score:.4f}")
                return score
        except Exception as e:
            logger.error(f"Fout bij CNN-voorspelling voor '{pattern_name}': {e}")
            return 0.0

    # --- Helperfuncties voor dataverwerking (uitbreiding van Freqtrade DF naar 'candles' dicts) ---
    def _dataframe_to_candles(self, dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Converteert een Freqtrade DataFrame naar een lijst van 'candle'-dictionaries.
        Dit is nodig omdat sommige JS detectiefuncties werkten met dictionaries.
        Freqtrade OHLCV is: timestamp, open, high, low, close, volume.
        """
        # Controleer of 'date' een kolom is, zo niet, probeer de index te resetten
        df_copy = dataframe.copy() # Werk op een kopie om de originele dataframe niet te wijzigen
        if 'date' not in df_copy.columns:
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index()
                # Controleer of 'index' de naam is na resetten, of 'date' als de index een naam had
                if 'index' in df_copy.columns and 'date' not in df_copy.columns:
                     df_copy.rename(columns={'index': 'date'}, inplace=True)
                elif df_copy.index.name == 'date' and 'date' not in df_copy.columns: # Als de index 'date' heette
                    df_copy.rename(columns={df_copy.index.name: 'date'}, inplace=True)


            else:
                logger.error("DataFrame heeft geen 'date' kolom of DatetimeIndex.")
                return []

        # Zorg ervoor dat de 'date' kolom datetime objecten bevat als het nog strings zijn
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            try:
                df_copy['date'] = pd.to_datetime(df_copy['date'])
            except Exception as e:
                logger.error(f"Kan 'date' kolom niet converteren naar datetime: {e}")
                return []


        candles = []
        for _, row in df_copy.iterrows():
            # Pas aan naar de daadwerkelijke kolomnamen in je Freqtrade DataFrame
            candle = {
                'time': row['date'].timestamp() * 1000, # Converteer naar ms voor consistentie met JS 'time'
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'quoteVolume': float(row.get('quote_volume', 0.0)), # Gebruik .get voor optionele kolommen
                'rsi': float(row.get('rsi', np.nan)),
                'macd': float(row.get('macd', np.nan)),
                'macdsignal': float(row.get('macdsignal', np.nan)),
                'macdhist': float(row.get('macdhist', np.nan)),
                'bb_upperband': float(row.get('bb_upperband', np.nan)),
                'bb_middleband': float(row.get('bb_middleband', np.nan)),
                'bb_lowerband': float(row.get('bb_lowerband', np.nan)),
            }
            candles.append(candle)
        return candles

    # --- Helperfunctie voor gemiddelde (uit reflectieAnalyser.js) ---
    def _mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        # Filter NaN waarden uit voordat gemiddelde wordt berekend
        valid_values = [v for v in values if not np.isnan(v)]
        if not valid_values:
            return 0.0
        return sum(valid_values) / len(valid_values)

    # --- Candlestick Patroon Detectie Functies (vertaald van JS) ---
    def detect_bull_flag(self, candles: List[Dict[str, Any]]) -> bool:
        if len(candles) < 10: return False
        # Vereenvoudigde logica, echte detectie is complexer en vereist trendlijnen
        # Zoek naar een sterke uptrend (flagpole) gevolgd door consolidatie (flag)
        pole_candles = candles[-10:-5] # Voorbeeld: eerste 5 candles van 10
        flag_candles = candles[-5:]    # Voorbeeld: laatste 5 candles van 10

        # Controleer flagpole (sterke uptrend)
        if not all(c['close'] > c['open'] for c in pole_candles if c is not None): # Check for None
            return False # Geen sterke bullish flagpole

        # Controleer flag (consolidatie of lichte daling)
        # Gemiddelde daling in de flag should zijn klein en tegen de trend
        flag_closes = np.array([c['close'] for c in flag_candles if c is not None])
        if len(flag_closes) < 2 : return False # Need at least 2 points for comparison
        if flag_closes[-1] < flag_closes[0] and (flag_closes[0] - flag_closes[-1]) / (flag_closes[0] + 1e-9) < 0.03: # Lichte daling <3%
            return True
        return False

    def detect_double_top(self, candles: List[Dict[str, Any]]) -> bool:
        if len(candles) < 20: return False
        # Simpele logica: twee pieken op ongeveer hetzelfde niveau, gescheiden door een dal
        highs = np.array([c['high'] for c in candles[-20:]])

        # Vereenvoudigde piekdetectie
        # Zoek naar twee relatieve maxima gescheiden door een relatief minimum
        if len(highs) < 20 : return False # Ensure enough data points

        # Split into two halves for peak detection
        first_half_highs = highs[:len(highs)//2]
        second_half_highs = highs[len(highs)//2:]

        if not first_half_highs.size or not second_half_highs.size: return False


        peak1_idx = np.argmax(first_half_highs)
        peak2_idx = np.argmax(second_half_highs) + len(highs)//2


        # Valideer dat de pieken op ongeveer hetzelfde niveau zijn
        if abs(highs[peak1_idx] - highs[peak2_idx]) / (highs[peak1_idx] + 1e-9) < 0.01: # Binnen 1% van elkaar
            # Valideer dat er een significant dal is tussen de pieken
            if peak1_idx < peak2_idx and min(highs[peak1_idx:peak2_idx]) < highs[peak1_idx] * 0.95: # Dal minstens 5% lager
                return True
        return False

    def detect_head_and_shoulders(self, candles: List[Dict[str, Any]]) -> bool:
        """
        Detecteert een Head and Shoulders patroon.
        Placeholder: Deze functie is nog niet volledig geïmplementeerd.
        """
        logger.warning(f"{self.__class__.__name__}.detect_head_and_shoulders is a placeholder and not fully implemented.")
        if len(candles) < 30: return False
        # Zeer complexe detectie die geavanceerdere piek-dal analyse vereist.
        # Voor nu een placeholder.
        return False

    def detect_breakout(self, candles: List[Dict[str, Any]], resistance_level: Optional[float] = None) -> bool:
        if len(candles) < 2: return False
        last_candle = candles[-1]

        # Als geen resistance_level is gegeven, bereken deze op basis van recente highs
        if resistance_level is None:
            prev_highs = [c['high'] for c in candles[-10:-1] if c is not None] # Vorige 9 highs
            if not prev_highs: return False
            resistance_level = max(prev_highs)

        # Een breakout treedt op wanneer de sluitingsprijs boven de weerstand ligt en er een significant volume is
        current_volume = last_candle.get('volume', 0)
        avg_volume_recent = self._mean([c.get('volume', 0) for c in candles[-20:-1] if c is not None]) # Gemiddelde van vorige 19 volumes

        return last_candle['close'] > resistance_level and current_volume > avg_volume_recent * 1.5

    def detect_ema_cross(self, candles: List[Dict[str, Any]], short_period: int = 12, long_period: int = 26) -> str or bool:
        if len(candles) < max(short_period, long_period): return False
        closes = np.array([c['close'] for c in candles if c is not None])
        if len(closes) < max(short_period, long_period): return False


        # Bereken EMA's met Pandas EWM (Exponential Weighted Mean)
        ema_short = pd.Series(closes).ewm(span=short_period, adjust=False).mean().iloc[-1]
        ema_long = pd.Series(closes).ewm(span=long_period, adjust=False).mean().iloc[-1]

        prev_closes = np.array([c['close'] for c in candles[:-1] if c is not None])
        if len(prev_closes) < max(short_period, long_period): return False # Zorg voor voldoende data voor vorige EMA
        prev_ema_short = pd.Series(prev_closes).ewm(span=short_period, adjust=False).mean().iloc[-1]
        prev_ema_long = pd.Series(prev_closes).ewm(span=long_period, adjust=False).mean().iloc[-1]

        if ema_short > ema_long and prev_ema_short <= prev_ema_long:
            return "bullishCross"
        if ema_short < ema_long and prev_ema_short >= prev_ema_long:
            return "bearishCross"
        return False

    def detect_cup_and_handle(self, candles: List[Dict[str, Any]]) -> bool:
        """
        Detecteert een Cup and Handle patroon.
        Placeholder: Deze functie is nog niet volledig geïmplementeerd.
        """
        logger.warning(f"{self.__class__.__name__}.detect_cup_and_handle is a placeholder and not fully implemented.")
        if len(candles) < 50: return False
        # Zeer complexe detectie, vereist geavanceerde vormherkenning. Placeholder.
        return False

    def detect_wedge_patterns(self, candles: List[Dict[str, Any]]) -> bool:
        """
        Detecteert Wedge (wig) patronen.
        Placeholder: Deze functie is nog niet volledig geïmplementeerd.
        """
        logger.warning(f"{self.__class__.__name__}.detect_wedge_patterns is a placeholder and not fully implemented.")
        if len(candles) < 20: return False
        # Vereenvoudigde logica voor dalende/stijgende wiggen (convergerende trendlijnen)
        # Vereist het fitten van trendlijnen. Placeholder.
        return False

    def detect_triple_top_bottom(self, candles: List[Dict[str, Any]]) -> bool:
        """
        Detecteert Triple Top of Triple Bottom patronen.
        Placeholder: Deze functie is nog niet volledig geïmplementeerd.
        """
        logger.warning(f"{self.__class__.__name__}.detect_triple_top_bottom is a placeholder and not fully implemented.")
        if len(candles) < 30: return False
        # Zeer complexe detectie. Placeholder.
        return False

    # --- V3 Candlestick Patroon Detectie (met TA-Lib waar mogelijk) ---
    def detect_candlestick_patterns(self, candles_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Detecteert klassieke candlestick patronen met behulp van TA-Lib.
        De input moet een Freqtrade-compatibel DataFrame zijn.
        """
        patterns = {}
        # Zorg ervoor dat de dataframe de juiste kolomnamen heeft voor TA-Lib
        # En dat er genoeg data is (TA-Lib functies kunnen crashen op te weinig data)
        if len(candles_df) < 1: # Sommige patronen vereisen meer, maar minstens 1
            return {}

        open_ = candles_df['open'].astype(float)
        high_ = candles_df['high'].astype(float)
        low_ = candles_df['low'].astype(float)
        close_ = candles_df['close'].astype(float)


        # Voorbeelden van TA-Lib candlestick patronen. De output is 0 of 100/-100, dus check op != 0
        # Gebruik try-except blokken voor elke TA-Lib call als extra voorzorg
        try: patterns['CDLDOJI'] = bool(talib.CDLDOJI(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLDOJI.lookback else False
        except Exception: patterns['CDLDOJI'] = False
        try: patterns['CDLHAMMER'] = bool(talib.CDLHAMMER(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLHAMMER.lookback else False
        except Exception: patterns['CDLHAMMER'] = False
        try: patterns['CDLINVERTEDHAMMER'] = bool(talib.CDLINVERTEDHAMMER(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLINVERTEDHAMMER.lookback else False
        except Exception: patterns['CDLINVERTEDHAMMER'] = False
        try: patterns['CDLENGULFING'] = bool(talib.CDLENGULFING(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLENGULFING.lookback else False
        except Exception: patterns['CDLENGULFING'] = False
        try: patterns['CDLMORNINGSTAR'] = bool(talib.CDLMORNINGSTAR(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLMORNINGSTAR.lookback else False
        except Exception: patterns['CDLMORNINGSTAR'] = False
        try: patterns['CDLEVENINGSTAR'] = bool(talib.CDLEVENINGSTAR(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLEVENINGSTAR.lookback else False
        except Exception: patterns['CDLEVENINGSTAR'] = False
        try: patterns['CDLHANGINGMAN'] = bool(talib.CDLHANGINGMAN(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLHANGINGMAN.lookback else False
        except Exception: patterns['CDLHANGINGMAN'] = False
        try: patterns['CDL3WHITESOLDIERS'] = bool(talib.CDL3WHITESOLDIERS(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDL3WHITESOLDIERS.lookback else False
        except Exception: patterns['CDL3WHITESOLDIERS'] = False
        try: patterns['CDL3BLACKCROWS'] = bool(talib.CDL3BLACKCROWS(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDL3BLACKCROWS.lookback else False
        except Exception: patterns['CDL3BLACKCROWS'] = False
        try: patterns['CDLPIERCING'] = bool(talib.CDLPIERCING(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLPIERCING.lookback else False
        except Exception: patterns['CDLPIERCING'] = False
        try: patterns['CDLDARKCLOUDCOVER'] = bool(talib.CDLDARKCLOUDCOVER(open_, high_, low_, close_).iloc[-1] != 0) if len(candles_df) >= talib.CDLDARKCLOUDCOVER.lookback else False
        except Exception: patterns['CDLDARKCLOUDCOVER'] = False


        # Filter op degenen die True zijn
        return {k: v for k, v in patterns.items() if v is True}

    # --- Aangepaste implementaties voor specifieke patronen die niet direct in TA-Lib zitten of aangepaste logica vereisen ---
    def _detect_engulfing(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 2: return False
        prev = candles[-2]
        curr = candles[-1]
        if prev is None or curr is None: return False
        bullish = prev['close'] < prev['open'] and curr['close'] > curr['open'] and                   curr['close'] > prev['open'] and curr['open'] < prev['close']
        bearish = prev['close'] > prev['open'] and curr['close'] < curr['open'] and                   curr['open'] > prev['close'] and curr['close'] < prev['open']
        if bullish: return "bullishEngulfing"
        if bearish: return "bearishEngulfing"
        return False

    def _detect_morning_evening_star(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 3: return False
        a, b, c = candles[-3:]
        if None in [a,b,c]: return False
        def body(x): return abs(x['close'] - x['open'])
        # Check for division by zero if body(a) is very small
        body_a = body(a)
        if body_a < 1e-9 : return False # Avoid division by zero or meaningless comparison

        bullish = a['close'] < a['open'] and body(b) < (body_a / 2) and                   c['close'] > c['open'] and c['close'] > ((a['open'] + a['close']) / 2)
        bearish = a['close'] > a['open'] and body(b) < (body_a / 2) and                   c['close'] < c['open'] and c['close'] < ((a['open'] + a['close']) / 2)
        return "morningStar" if bullish else ("eveningStar" if bearish else False)

    def _detect_triangle(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 10: return False
        highs = [c['high'] for c in candles[-10:] if c is not None]
        lows = [c['low'] for c in candles[-10:] if c is not None]
        if len(highs) < 10 or len(lows) < 10: return False # Not enough valid data

        # Vereenvoudigde detectie: convergentie van highs en lows
        # Controleer op de trend van highs en lows
        avg_high_start = self._mean(highs[:5])
        avg_high_end = self._mean(highs[-5:])
        avg_low_start = self._mean(lows[:5])
        avg_low_end = self._mean(lows[-5:])

        # Ascending triangle (vlakke top, stijgende bodem)
        if (abs(avg_high_end - avg_high_start) / (avg_high_start + 1e-9) < 0.01 and # Relatief vlakke highs
            avg_low_end > avg_low_start * 1.01): # Stijgende lows
            return "ascendingTriangle"

        # Descending triangle (dalende top, vlakke bodem)
        if (avg_high_end < avg_high_start * 0.99 and # Dalende highs
            abs(avg_low_end - avg_low_start) / (avg_low_start + 1e-9) < 0.01): # Relatief vlakke lows
            return "descendingTriangle"

        # Symmetrical triangle (convergerende highs en lows)
        if (avg_high_end < avg_high_start * 0.99 and avg_low_end > avg_low_start * 1.01 and
            (max(highs) - min(lows)) / (self._mean(highs) + 1e-9) > 0.02): # Zorg dat er nog een significant bereik is
            return "symmetricalTriangle"
        return False

    def _detect_channel(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 10: return False
        highs = [c['high'] for c in candles[-10:] if c is not None]
        lows = [c['low'] for c in candles[-10:] if c is not None]
        if len(highs) < 10 or len(lows) < 10: return False

        # Check voor parallelle beweging tussen highs en lows
        high_range = max(highs) - min(highs)
        low_range = max(lows) - min(lows)
        mean_highs = self._mean(highs)
        mean_lows = self._mean(lows)


        # Als de ranges klein zijn en de afstand tussen highs en lows consistent is
        if mean_highs == 0 or mean_lows == 0 : return False # Avoid division by zero
        if high_range / mean_highs < 0.02 and low_range / mean_lows < 0.02:
            # Check voor consistentie in de breedte van het kanaal
            channel_widths = [h - l for h,l in zip(highs, lows)]
            if not channel_widths: return False
            avg_channel_width_start = self._mean(channel_widths[:5])
            avg_channel_width_end = self._mean(channel_widths[-5:])

            if avg_channel_width_start == 0 : return False # Avoid division by zero
            if abs(avg_channel_width_start - avg_channel_width_end) / avg_channel_width_start < 0.1: # Binnen 10% consistent
                # Bepaal de richting van het kanaal
                if highs[-1] > highs[0] and lows[-1] > lows[0]: return "upChannel"
                if highs[-1] < highs[0] and lows[-1] < lows[0]: return "downChannel"
                return "horizontalChannel"
        return False

    def _detect_pennant(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 15: return False
        # Een pennant volgt vaak een sterke beweging (flagpole) en dan een consolidatie in een driehoek
        # Eenvoudige benadering: sterke beweging, gevolgd door samentrekkend bereik

        # Check for flagpole (initial strong move) - e.g., last 5 candles for strength
        flagpole_candles = [c for c in candles[-15:-5] if c is not None]
        if len(flagpole_candles) < 2 : return False

        initial_open = flagpole_candles[0]['open']
        if initial_open == 0 : return False # Avoid division by zero
        initial_move_perc = abs(flagpole_candles[-1]['close'] - initial_open) / initial_open
        if initial_move_perc < 0.05: return False # Require at least 5% initial move


        # Check for consolidating triangle (pennant body) in the last 5-10 candles
        pennant_candles = [c for c in candles[-5:] if c is not None]
        if len(pennant_candles) < 2: return False

        highs_pennant = [c['high'] for c in pennant_candles]
        lows_pennant = [c['low'] for c in pennant_candles]

        # Check for converging highs and lows within the pennant body
        # Ensure highs and lows are not empty
        if not highs_pennant or not lows_pennant: return False

        range_start = highs_pennant[0] - lows_pennant[0] # Initial range in pennant
        range_end = highs_pennant[-1] - lows_pennant[-1] # Final range in pennant

        if range_start == 0 : return False # Avoid division by zero
        if range_end < range_start * 0.7: # Significant range contraction (e.g., last range is < 70% of first)
            return "pennant"
        return False

    def _detect_parabolic_curve(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 10: return False
        # Check voor een steeds steilere stijging (versnelling)
        closes = np.array([c['close'] for c in candles[-10:] if c is not None])
        if len(closes) < 3: return False

        accelerations = []
        for i in range(2, len(closes)):
            # Tweede afgeleide benadering: verandering in verandering van prijs
            accel = (closes[i] - closes[i-1]) - (closes[i-1] - closes[i-2])
            accelerations.append(accel)

        if not accelerations: return False

        # Als de meeste acceleraties positief zijn en significant (voor bullish curve)
        # Of negatief en significant (voor bearish curve)
        avg_accel = np.mean(accelerations)
        if closes[-1] == 0 : return False # Avoid division by zero
        if avg_accel > 0.005 * closes[-1] and all(a > 0 for a in accelerations[-3:]): # laatste 3 zijn positief
            return "parabolicCurveUp"
        if avg_accel < -0.005 * closes[-1] and all(a < 0 for a in accelerations[-3:]): # laatste 3 zijn negatief
            return "parabolicCurveDown"

        return False

    def _detect_hanging_man(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 1: return False
        c = candles[-1]
        if c is None: return False
        body = abs(c['close'] - c['open'])
        lower_shadow = min(c['open'], c['close']) - c['low']
        total_range = c['high'] - c['low']
        # Hanging Man heeft een klein lichaam aan de bovenkant van de candle
        # en een lange onderschaduw, en een kleine of geen bovenschaduw.
        # Moet verschijnen na een uptrend (impliciet hier).
        if total_range == 0 : return False # Avoid division by zero
        if (lower_shadow > body * 2 and # Onderste schaduw minstens twee keer het lichaam
            body < total_range * 0.3 and # Lichaam is klein (minder dan 30% van de totale range)
            (c['high'] - max(c['open'], c['close'])) < body): # Kleine of geen bovenschaduw
            return "hangingMan"
        return False

    def _detect_inverted_hammer(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 1: return False
        c = candles[-1]
        if c is None: return False
        body = abs(c['close'] - c['open'])
        upper_shadow = c['high'] - max(c['open'], c['close'])
        total_range = c['high'] - c['low']

        # Inverted Hammer heeft een klein lichaam aan de onderkant van de candle
        # en een lange bovenschaduw, en een kleine of geen onderschaduw.
        # Moet verschijnen na een downtrend (impliciet hier).
        if total_range == 0 : return False
        if (upper_shadow > body * 2 and # Bovenste schaduw minstens twee keer het lichaam
            body < total_range * 0.3 and # Lichaam is klein
            (min(c['open'], c['close']) - c['low']) < body): # Kleine of geen onderschaduw
            return "invertedHammer"
        return False

    def _detect_three_white_soldiers(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 3: return False
        a, b, c = candles[-3:]
        if None in [a,b,c]: return False
        # Drie opeenvolgende lange bullish candles.
        # Elk opent binnen/nabij het lichaam van de vorige en sluit hoger, met weinig schaduwen.
        cond1 = a['close'] > a['open'] and b['close'] > b['open'] and c['close'] > c['open'] # Allemaal bullish
        cond2 = b['open'] > a['open'] and c['open'] > b['open'] # Elke candle opent binnen of boven het lichaam van de vorige
        cond3 = b['close'] > a['close'] and c['close'] > b['close'] # Elke candle sluit hoger dan de vorige

        # Controleer op relatief lange lichamen en kleine schaduwen
        # Avoid division by zero if high == low
        body_a_perc = abs(a['close'] - a['open']) / (a['high'] - a['low'] + 1e-9)
        body_b_perc = abs(b['close'] - b['open']) / (b['high'] - b['low'] + 1e-9)
        body_c_perc = abs(c['close'] - c['open']) / (c['high'] - c['low'] + 1e-9)


        if cond1 and cond2 and cond3 and body_a_perc > 0.6 and body_b_perc > 0.6 and body_c_perc > 0.6:
            return "threeWhiteSoldiers"
        return False

    def _detect_three_black_crows(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 3: return False
        a, b, c = candles[-3:]
        if None in [a,b,c]: return False
        # Drie opeenvolgende lange bearish candles.
        # Elk opent binnen/nabij het lichaam van de vorige en sluit lager.
        cond1 = a['close'] < a['open'] and b['close'] < b['open'] and c['close'] < c['open'] # Allemaal bearish
        cond2 = b['open'] < a['open'] and c['open'] < b['open'] # Elke candle opent binnen of onder het lichaam van de vorige
        cond3 = b['close'] < a['close'] and c['close'] < b['close'] # Elke candle sluit lager dan de vorige

        body_a_perc = abs(a['close'] - a['open']) / (a['high'] - a['low'] + 1e-9)
        body_b_perc = abs(b['close'] - b['open']) / (b['high'] - b['low'] + 1e-9)
        body_c_perc = abs(c['close'] - c['open']) / (c['high'] - c['low'] + 1e-9)

        if cond1 and cond2 and cond3 and body_a_perc > 0.6 and body_b_perc > 0.6 and body_c_perc > 0.6:
            return "threeBlackCrows"
        return False

    def _detect_piercing_line(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 2: return False
        a, b = candles[-2:]
        if None in [a,b]: return False
        # Bearish candle gevolgd door bullish die meer dan 50% van de vorige sluit
        midpoint_a = (a['open'] + a['close']) / 2
        if a['close'] < a['open'] and b['close'] > b['open'] and            b['open'] < a['close'] and b['close'] > midpoint_a:
            return "piercingLine"
        return False

    def _detect_dark_cloud_cover(self, candles: List[Dict[str, Any]]) -> str or bool:
        if len(candles) < 2: return False
        a, b = candles[-2:]
        if None in [a,b]: return False
        # Bullish candle gevolgd door bearish die meer dan 50% van de vorige sluit
        midpoint_a = (a['open'] + a['close']) / 2
        if a['open'] < a['close'] and b['close'] < b['open'] and            b['open'] > a['close'] and b['close'] < midpoint_a:
            return "darkCloudCover"
        return False

    # --- Indicator Patroon Detectie Functies ---

    def detect_rsi_divergence(self, candles: List[Dict[str, Any]]) -> bool:
        # Vereist RSI en prijsactie analyse. Meer geavanceerde logica dan simpele vergelijking.
        if len(candles) < 30: return False # Genoeg data voor trends

        closes = np.array([c.get('close', np.nan) for c in candles if c is not None])
        rsi_values = np.array([c.get('rsi', np.nan) for c in candles if c is not None])


        # Verwijder NaN waarden
        valid_indices = ~np.isnan(closes) & ~np.isnan(rsi_values)
        closes = closes[valid_indices]
        rsi_values = rsi_values[valid_indices]

        if len(closes) < 10 or len(rsi_values) < 10: return False # Niet genoeg geldige data na filtering

        # Simpele bearish divergentie: Prijs maakt hogere highs, RSI maakt lagere highs
        # Zoek naar lokale pieken (vereenvoudigd)
        # Dit is een zeer basale piekdetectie, kan verbeterd worden met scipy.signal.find_peaks

        # Voor nu, laten we kijken naar de laatste N punten
        N = 10 # Kijk naar de laatste 10 punten voor divergentie
        if len(closes) < N or len(rsi_values) < N: return False

        recent_closes = closes[-N:]
        recent_rsi = rsi_values[-N:]

        # Bearish: Prijs stijgt, RSI daalt
        if recent_closes[-1] > recent_closes[0] and recent_rsi[-1] < recent_rsi[0]:
             # Check of de trend in prijs consistent is (e.g. geen grote dip in het midden)
            price_trend_consistent = all(recent_closes[i] >= recent_closes[i-1] for i in range(1,N//2)) and                                      all(recent_closes[i] >= recent_closes[i-1] for i in range(N//2,N))
            rsi_trend_consistent = all(recent_rsi[i] <= recent_rsi[i-1] for i in range(1,N//2)) and                                    all(recent_rsi[i] <= recent_rsi[i-1] for i in range(N//2,N))
            if price_trend_consistent and rsi_trend_consistent:
                return "bearishRSIDivergence"


        # Bullish: Prijs daalt, RSI stijgt
        if recent_closes[-1] < recent_closes[0] and recent_rsi[-1] > recent_rsi[0]:
            price_trend_consistent = all(recent_closes[i] <= recent_closes[i-1] for i in range(1,N//2)) and                                      all(recent_closes[i] <= recent_closes[i-1] for i in range(N//2,N))
            rsi_trend_consistent = all(recent_rsi[i] >= recent_rsi[i-1] for i in range(1,N//2)) and                                    all(recent_rsi[i] >= recent_rsi[i-1] for i in range(N//2,N))
            if price_trend_consistent and rsi_trend_consistent:
                return "bullishRSIDivergence"


        return False


    def detect_macd_divergence(self, candles: List[Dict[str, Any]]) -> bool:
        """
        Detecteert MACD divergentie.
        Let op: Huidige implementatie is een vereenvoudiging en mogelijk niet volledig robuust.
        """
        # logger.warning(f"{self.__class__.__name__}.detect_macd_divergence is a simplified implementation.") # Optional: if you want to warn on every call
        if len(candles) < 50: return False

        closes = np.array([c.get('close', np.nan) for c in candles if c is not None])
        macd_hist_values = np.array([c.get('macdhist', np.nan) for c in candles if c is not None])

        valid_indices = ~np.isnan(closes) & ~np.isnan(macd_hist_values)
        closes = closes[valid_indices]
        macd_hist_values = macd_hist_values[valid_indices]

        if len(closes) < 20 or len(macd_hist_values) < 20: return False # Niet genoeg geldige data

        # Simpele bearish divergentie: Prijs HH, MACD Hist LH
        # Simpele bullish divergentie: Prijs LL, MACD Hist HL
        # Dit vereist piek/dal detectie op zowel prijs als MACD-histogram
        # Voor nu, een simpele placeholder over de laatste 5-10 candles:
        N = 10
        if len(closes) < N or len(macd_hist_values) < N : return False
        recent_closes = closes[-N:]
        recent_macd_hist = macd_hist_values[-N:]

        if recent_closes[-1] > recent_closes[0] and recent_macd_hist[-1] < recent_macd_hist[0] and recent_macd_hist[-1] > 0 and recent_macd_hist[0] > 0:
            return "bearishMACDDivergence"
        if recent_closes[-1] < recent_closes[0] and recent_macd_hist[-1] > recent_macd_hist[0] and recent_macd_hist[-1] < 0 and recent_macd_hist[0] < 0:
            return "bullishMACDDivergence"
        return False

    def detect_ichimoku_twist(self, candles: List[Dict[str, Any]]) -> bool:
        """
        Detecteert een Ichimoku Kumo (cloud) twist.
        Placeholder: Deze functie is nog niet volledig geïmplementeerd.
        """
        logger.warning(f"{self.__class__.__name__}.detect_ichimoku_twist is a placeholder and not fully implemented.")
        # Vereist uitgebreide Ichimoku berekeningen en analyse van Kumo (cloud) kruisingen.
        # Complexe logica, placeholder.
        return False

    def detect_bollinger_squeeze(self, candles: List[Dict[str, Any]]) -> bool:
        if len(candles) < 20: return False # Nodig voor 20-periode BB

        upper_band = np.array([c.get('bb_upperband', np.nan) for c in candles if c is not None])
        lower_band = np.array([c.get('bb_lowerband', np.nan) for c in candles if c is not None])

        valid_indices = ~np.isnan(upper_band) & ~np.isnan(lower_band)
        upper_band = upper_band[valid_indices]
        lower_band = lower_band[valid_indices]

        if len(upper_band) < 20 or len(lower_band) < 20 : return False

        # Bereken de bandbreedte
        band_width = upper_band - lower_band
        if len(band_width) < 2 : return False # Need at least 2 points for percentile


        # Controleer of de bandbreedte significant vernauwt ten opzichte van het verleden
        # Een squeeze is typisch wanneer de bandbreedte een recente low bereikt
        if band_width[-1] < np.percentile(band_width[:-1], 20): # Als huidige breedte in de onderste 20% van recente breedtes zit
            return "bollingerSqueeze"
        return False

    def detect_vwap_cross(self, candles: List[Dict[str, Any]]) -> str or bool:
        """
        Detecteert een kruising met de Volume Weighted Average Price (VWAP).
        Placeholder: Deze functie is nog niet volledig geïmplementeerd (vereist VWAP data).
        """
        logger.warning(f"{self.__class__.__name__}.detect_vwap_cross is a placeholder and not fully implemented (requires VWAP data).")
        # VWAP berekening is complexer dan simpele gemiddelden. Vereist cumulatieve Price * Volume.
        # Freqtrade kan VWAP berekenen via custom indicators, dan kan je hier op checken.
        # Placeholder.
        return False

    def detect_heikin_ashi_trend(self, candles: List[Dict[str, Any]]) -> str or bool:
        # Eerst omzetten naar Heikin-Ashi candles
        if len(candles) < 5: return False # Minimaal aantal candles voor trend

        valid_candles = [c for c in candles if c is not None]
        if len(valid_candles) < 5 : return False


        # Simulate conversion to Heikin Ashi
        ha_candles = []
        for i in range(len(valid_candles)):
            c = valid_candles[i]
            ha_close = (c['open'] + c['high'] + c['low'] + c['close']) / 4
            # Corrected ha_open logic for first candle
            ha_open = (valid_candles[i-1]['open'] + valid_candles[i-1]['close']) / 2 if i > 0 else (c['open'] + c['close']) / 2
            if i > 0: # For subsequent candles, use previous HA open and HA close
                 ha_open = (ha_candles[i-1]['open'] + ha_candles[i-1]['close']) / 2

            ha_high = np.max([c['high'], ha_open, ha_close])
            ha_low = np.min([c['low'], ha_open, ha_close])
            ha_candles.append({'open': ha_open, 'close': ha_close, 'high': ha_high, 'low': ha_low})

        # Controleer trend van de laatste paar Heikin-Ashi candles
        if len(ha_candles) < 3 : return False
        last_ha_candles = ha_candles[-3:] # Neem de laatste 3

        uptrend = all(hc['close'] > hc['open'] for hc in last_ha_candles)
        downtrend = all(hc['close'] < hc['open'] for hc in last_ha_candles)

        if uptrend and all(last_ha_candles[i]['open'] >= last_ha_candles[i-1]['open'] for i in range(1, len(last_ha_candles))): # Consistent oplopende opening
            return "heikinUptrend"
        if downtrend and all(last_ha_candles[i]['open'] <= last_ha_candles[i-1]['open'] for i in range(1, len(last_ha_candles))): # Consistent dalende opening
            return "heikinDowntrend"


        return False

    def detect_fractals(self, candles: List[Dict[str, Any]]) -> str or bool:
        # Detectie van fractals (swing highs/lows)
        if len(candles) < 5: return False

        valid_candles = [c for c in candles if c is not None]
        if len(valid_candles) < 5 : return False


        # Een bullish fractal is een patroon van 5 bars waar de middelste bar de hoogste high heeft.
        # Een bearish fractal is een patroon van 5 bars waar de middelste bar de laagste low heeft.

        # Focus on the set of 5 candles ending at the most recent available full set
        # Example: if len is 7, we check candles[2] to candles[6] (indices), focusing on candles[4] as mid
        # For the LAST possible fractal, mid_idx is len(candles) - 3

        mid_idx = len(valid_candles) - 3 # De middelste candle in de laatste 5
        if mid_idx < 2: return False # We need 2 candles on each side of mid

        c_mid = valid_candles[mid_idx]
        c_l2 = valid_candles[mid_idx - 2]
        c_l1 = valid_candles[mid_idx - 1]
        c_r1 = valid_candles[mid_idx + 1]
        c_r2 = valid_candles[mid_idx + 2]


        is_bullish_fractal = (c_mid['high'] > c_l2['high'] and
                              c_mid['high'] > c_l1['high'] and
                              c_mid['high'] > c_r1['high'] and
                              c_mid['high'] > c_r2['high'])

        is_bearish_fractal = (c_mid['low'] < c_l2['low'] and
                              c_mid['low'] < c_l1['low'] and
                              c_mid['low'] < c_r1['low'] and
                              c_mid['low'] < c_r2['low'])

        if is_bullish_fractal: return "bullishFractal"
        if is_bearish_fractal: return "bearishFractal"

        return False

    # --- Contextuele Detectie Functies ---

    def determine_trend(self, candles: List[Dict[str, Any]]) -> str:
        if len(candles) < 20: return "unknown"
        closes = np.array([c['close'] for c in candles if c is not None])
        if len(closes) < 20 : return "unknown"
        # Eenvoudige trend bepaling: vergelijking van gemiddelden
        avg_short = np.mean(closes[-5:])
        avg_long = np.mean(closes[-20:])
        if avg_long == 0 : return "unknown" # Avoid division by zero
        if avg_short > avg_long * 1.01: return "uptrend"
        if avg_short < avg_long * 0.99: return "downtrend"
        return "sideways"

    def detect_volume_spike(self, candles: List[Dict[str, Any]], multiplier: float = 2.0) -> bool:
        if len(candles) < 20: return False
        volumes = np.array([c['volume'] for c in candles if c is not None])
        if len(volumes) < 20 : return False

        avg_volume = self._mean(volumes[:-1]) # Gemiddelde van vorige volumes
        if avg_volume == 0 : return False # Avoid issues if avg_volume is zero
        last_volume = volumes[-1]
        return last_volume > avg_volume * multiplier


    # --- Hoofddetectie Logica voor Multi-Timeframe ---

    async def detect_patterns_multi_timeframe(self, candles_by_timeframe: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """
        Detecteert patronen over meerdere timeframes en consolideert de resultaten.
        Vertaald en geoptimaliseerd van detectPatternsMultiTimeframe in cnnPatternRecognizer-uitgebreid.js.
        """
        all_patterns = {
            "zoomPatterns": {},
            "contextPatterns": {},
            "patterns": {}, # For rule-based patterns
            "cnn_predictions": {}, # For CNN model predictions
            "context": {"trend": "unknown", "volume_spike": False}
        }

        TIME_FRAME_CONFIG = {
            "zoom": ['1m', '5m', '15m'],
            "context": ['1h', '4h', '12h', '1d', '1w']
        }

        # --- Regelgebaseerde detectie (bestaande logica) ---
        available_timeframes = candles_by_timeframe.keys()
        for tf_group in TIME_FRAME_CONFIG.values():
            for tf_val in tf_group:
                 if tf_val not in available_timeframes:
                    logger.warning(f"Timeframe {tf_val} niet beschikbaar in candles_by_timeframe voor {symbol}. Patroondetectie kan onvolledig zijn.")


        # Detecteer patronen per timeframe
        for timeframe_type, timeframes in TIME_FRAME_CONFIG.items():
            for tf in timeframes:
                if tf in candles_by_timeframe and not candles_by_timeframe[tf].empty:
                    # Converteer Freqtrade DataFrame naar de verwachte lijst van dicts
                    candles_list = self._dataframe_to_candles(candles_by_timeframe[tf])

                    if not candles_list:
                        logger.warning(f"Geen geldige candlelijst voor {symbol} op {tf}. Overslaan patroondetectie voor dit TF.")
                        continue

                    # Detecteer de patronen
                    # Let op: detect_candlestick_patterns vereist dataframe, de rest lijst van dicts

                    # --- CNN Model Voorspelling (Integratie) ---
                    # Dit zal nu scores retourneren als het model geladen is
                    # candles_df is candles_by_timeframe[tf]
                    if not candles_by_timeframe[tf].empty:
                        for pattern_name_cnn in self.pattern_configs.keys(): # Loop over patronen waarvoor we modellen hebben
                            score = await self.predict_pattern_score(candles_by_timeframe[tf], pattern_name_cnn)
                            if score > 0.0: # Only add if a prediction was made (model loaded and score is positive)
                                all_patterns["cnn_predictions"][f"{tf}_{pattern_name_cnn}_score"] = score

                    detected_tf_patterns = {
                        "bullFlag": self.detect_bull_flag(candles_list),
                        "doubleTop": self.detect_double_top(candles_list),
                        "headAndShoulders": self.detect_head_and_shoulders(candles_list),
                        "breakout": self.detect_breakout(candles_list),
                        "emaCross": self.detect_ema_cross(candles_list),
                        "cupAndHandle": self.detect_cup_and_handle(candles_list),
                        "wedgePatterns": self.detect_wedge_patterns(candles_list),
                        "tripleTopBottom": self.detect_triple_top_bottom(candles_list),
                        "engulfing": self._detect_engulfing(candles_list),
                        "morningEveningStar": self._detect_morning_evening_star(candles_list),
                        "triangle": self._detect_triangle(candles_list),
                        "channel": self._detect_channel(candles_list),
                        "pennant": self._detect_pennant(candles_list),
                        "parabolicCurve": self._detect_parabolic_curve(candles_list),
                        "hangingMan": self._detect_hanging_man(candles_list),
                        "invertedHammer": self._detect_inverted_hammer(candles_list),
                        "threeWhiteSoldiers": self._detect_three_white_soldiers(candles_list),
                        "threeBlackCrows": self._detect_three_black_crows(candles_list),
                        "piercingLine": self._detect_piercing_line(candles_list),
                        "darkCloudCover": self._detect_dark_cloud_cover(candles_list),
                        "rsiDivergence": self.detect_rsi_divergence(candles_list),
                        "macdDivergence": self.detect_macd_divergence(candles_list),
                        "ichimokuTwist": self.detect_ichimoku_twist(candles_list),
                        "bollingerSqueeze": self.detect_bollinger_squeeze(candles_list),
                        "vwapCross": self.detect_vwap_cross(candles_list),
                        "heikinAshiTrend": self.detect_heikin_ashi_trend(candles_list),
                        "fractals": self.detect_fractals(candles_list),
                    }

                    # Voeg TA-Lib candlestick patronen toe
                    if not candles_by_timeframe[tf].empty:
                        talib_patterns = self.detect_candlestick_patterns(candles_by_timeframe[tf])
                        detected_tf_patterns.update(talib_patterns)
                    else:
                        logger.warning(f"DataFrame voor {symbol} op {tf} is leeg. Overslaan TA-Lib patroondetectie.")


                    # Filter alleen de True/gedetecteerde patronen (of non-False strings)
                    filtered_patterns = {k: v for k, v in detected_tf_patterns.items() if v}


                    if timeframe_type == "zoom":
                        all_patterns["zoomPatterns"][tf] = filtered_patterns
                    else: # context
                        all_patterns["contextPatterns"][tf] = filtered_patterns

        # Consolidatie van gevalideerde patronen (hier wordt de logica complexer)
        # Voorbeeld: een patroon is 'gevalideerd' als het op meerdere relevante timeframes verschijnt.
        # Dit is de plek waar je complexere ML-modellen (getrainde CNN's) de patroonherkenning zouden doen.
        # Voor nu een eenvoudige aggregatie:
        for tf_type_key in ["zoomPatterns", "contextPatterns"]:
            if tf_type_key in all_patterns:
                for tf, patterns_on_tf in all_patterns[tf_type_key].items():
                    for pattern_name, status in patterns_on_tf.items():
                        if status: # Als patroon gedetecteerd is
                            # Simpele logica: als het op een zoom of context TF wordt gezien, voeg het toe
                            # Je kunt hier complexere logica toevoegen, bijv. "bullFlag": "confirmed" als het op 5m en 1h is
                            all_patterns["patterns"][pattern_name] = status # Slaat de status direct op

        # Bepaal context (trend, volume spike) op basis van een langere timeframe (bijv. 4h of 1d)
        main_context_tf_options = ['1h', '4h', '1d']
        chosen_context_tf = None
        for tf_opt in main_context_tf_options:
            if tf_opt in candles_by_timeframe and not candles_by_timeframe[tf_opt].empty:
                chosen_context_tf = tf_opt
                break

        if chosen_context_tf:
            context_candles = self._dataframe_to_candles(candles_by_timeframe[chosen_context_tf])
            if context_candles: # Ensure list is not empty
                all_patterns["context"]["trend"] = self.determine_trend(context_candles)
                all_patterns["context"]["volume_spike"] = self.detect_volume_spike(context_candles)
            else:
                logger.warning(f"Context candle list voor {symbol} op {chosen_context_tf} is leeg na conversie.")
                all_patterns["context"]["trend"] = "onbekend"
                all_patterns["context"]["volume_spike"] = False

        else:
            logger.warning(f"Geen geschikte context timeframe ({', '.join(main_context_tf_options)}) beschikbaar voor trend/volume detectie voor {symbol}.")
            all_patterns["context"]["trend"] = "onbekend"
            all_patterns["context"]["volume_spike"] = False


        logger.debug(f"Patroondetectie resultaat voor {symbol}: {json.dumps(all_patterns, indent=2, default=str)}") # Added default=str
        return all_patterns

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import asyncio
    import pandas as pd
    from datetime import datetime, timedelta
    import dotenv
    import sys
    FREQTRADE_DB_PATH = "freqtrade_test.sqlite3"
    # Removed sqlite3 import and FREQTRADE_DB_PATH logic as it's not directly relevant to testing cnn_patterns.py unit logic
    # and might cause issues if the DB doesn't exist or path is incorrect in a generic test environment.
    # Focus will be on mock dataframes.

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Attempt to load .env from parent directory, common for project structures
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if os.path.exists(dotenv_path):
        dotenv.load_dotenv(dotenv_path)
        logger.info(f".env file loaded from {dotenv_path}")
    else:
        logger.info(".env file not found, proceeding without it (environment variables might be set externally).")

    # Create dummy CNN models for testing purposes (for bullFlag and bearishEngulfing)
    # Ensure MODELS_DIR is defined and accessible. It should be a class variable in CNNPatterns.
    # MODELS_DIR = CNNPatterns.MODELS_DIR # Access it if defined in class
    # If not, redefine for test or ensure CNNPatterns.MODELS_DIR is static or accessible.
    # For this subtask, assume MODELS_DIR is available via CNNPatterns.MODELS_DIR or defined globally in the script for testing.
    # The main CNNPatterns class should define MODELS_DIR.

    os.makedirs(CNNPatterns.MODELS_DIR, exist_ok=True) # Use class's MODELS_DIR

    def create_mock_dataframe_for_cnn(timeframe: str, num_rows: int) -> pd.DataFrame:
        # Create a mock DataFrame with necessary columns for CNN input
        data = {
            'date': pd.to_datetime([datetime.utcnow() - timedelta(minutes=i) for i in range(num_rows)])[::-1],
            'open': np.random.rand(num_rows) * 100,
            'high': np.random.rand(num_rows) * 100 + 100,
            'low': np.random.rand(num_rows) * 100 - 50,
            'close': np.random.rand(num_rows) * 100,
            'volume': np.random.rand(num_rows) * 1000,
            'rsi': np.random.rand(num_rows) * 100,
            'macd': np.random.rand(num_rows) * 10,
            'macdsignal': np.random.rand(num_rows) * 10,
            'macdhist': np.random.rand(num_rows) * 10,
            'bb_upperband': np.random.rand(num_rows) * 100 + 5,
            'bb_middleband': np.random.rand(num_rows) * 100,
            'bb_lowerband': np.random.rand(num_rows) * 100 - 5,
        }
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True) # Set 'date' as index like Freqtrade data
        return df

    async def run_test_cnn_patterns():
        cnn_detector = CNNPatterns() # This will now load models based on pattern_configs
        test_symbol_val = 'ETH/USDT' # Renamed variable
            'open': np.random.rand(num_rows) * 100,
            'high': np.random.rand(num_rows) * 100 + 100,
            'low': np.random.rand(num_rows) * 100 - 50,
            'close': np.random.rand(num_rows) * 100,
            'volume': np.random.rand(num_rows) * 1000,
            'rsi': np.random.rand(num_rows) * 100,
            'macd': np.random.rand(num_rows) * 10,
            'macdsignal': np.random.rand(num_rows) * 10,
            'macdhist': np.random.rand(num_rows) * 10,
            'bb_upperband': np.random.rand(num_rows) * 100 + 5,
            'bb_middleband': np.random.rand(num_rows) * 100,
            'bb_lowerband': np.random.rand(num_rows) * 100 - 5,
        }
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        # Ensure all required columns for _dataframe_to_cnn_input are present
        features_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist']
        for col in features_cols:
            if col not in df.columns:
                df[col] = np.nan # or some default value
        return df

    # Mock SimpleCNN model for saving/loading test (already defined in the file)
    # class MockSimpleCNN(nn.Module): # Not needed, SimpleCNN is defined above
    #     def __init__(self, input_channels, num_classes):
    #         super().__init__()
    #         self.linear = nn.Linear(input_channels * 30, num_classes)
    #     def forward(self, x):
    #         # Simulate a forward pass that works with the expected input shape
    #         # x shape: (batch, channels, sequence_length), e.g., (1, 9, 30)
    #         batch_size, channels, seq_len = x.shape
    #         # Flatten appropriately for a simple linear layer if that's the test model structure
    #         # x_flat = x.view(batch_size, -1) # Flatten all features
    #         # return self.linear(x_flat)
    #         # For SimpleCNN as defined, it has its own forward pass.
    #         # The dummy model saved just needs to match the structure for state_dict loading.
    #         return torch.randn(x.size(0), 2) # Return dummy logits

    # Create dummy model files and scaler params
    for pattern_name_config in ['bullFlag', 'bearishEngulfing']: # Use a different variable name
        # Access paths from a dummy config or CNNPatterns instance for consistency
        # Assuming cnn_detector is not yet instantiated, or access pattern_configs definition if needed
        # For simplicity, construct paths directly for the test setup.
        model_path_test = os.path.join(CNNPatterns.MODELS_DIR, f'cnn_model_{pattern_name_config}.pth')
        scaler_path_test = os.path.join(CNNPatterns.MODELS_DIR, f'scaler_params_{pattern_name_config}.json')

        if not os.path.exists(model_path_test):
            # Use the actual SimpleCNN definition for saving the dummy model
            dummy_model = SimpleCNN(input_channels=9, num_classes=2) # Matches pattern_configs
            torch.save(dummy_model.state_dict(), model_path_test)
            logger.info(f"Dummy CNN model for {pattern_name_config} saved at: {model_path_test}")

        if not os.path.exists(scaler_path_test):
            dummy_scaler_params = {
                'min_': np.array([0.0] * 9).tolist(), # Ensure it's list for JSON
                'scale_': np.array([1.0] * 9).tolist(), # Ensure it's list for JSON
                'feature_names_in_': ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist'] # For sklearn > 0.24
            }
            with open(scaler_path_test, 'w', encoding='utf-8') as f:
                json.dump(dummy_scaler_params, f)
            logger.info(f"Dummy scaler params for {pattern_name_config} saved at: {scaler_path_test}")

    # Mock Freqtrade DataFrames for different timeframes
    def create_mock_dataframe_for_cnn(timeframe_str: str, num_candles: int = 100) -> pd.DataFrame: # Renamed parameter
        data = []
        now_dt = datetime.utcnow() # Renamed variable
        interval_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
        interval_sec = interval_map.get(timeframe_str, 300)

        base_price = 100
        for i_val in range(num_candles): # Renamed variable
            date_val = now_dt - timedelta(seconds=(num_candles - 1 - i_val) * interval_sec) # Renamed variable
            o = base_price + i_val * 0.05 + np.random.uniform(-0.5, 0.5) # open
            c = o + np.random.uniform(-1, 1) # close
            h = max(o, c) + np.random.uniform(0, 0.5) # high
            l = min(o, c) - np.random.uniform(0, 0.5) # low
            v = np.random.uniform(10, 100) # volume

            # Mock indicators - ensure these match features_cols
            rsi = np.random.uniform(30, 70)
            macd = np.random.uniform(-0.1, 0.1)
            macdsignal = macd + np.random.uniform(-0.05, 0.05)
            macdhist = macd - macdsignal

            data.append([date_val, o, h, l, c, v, rsi, macd, macdsignal, macdhist])

        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist'])
        df['date'] = pd.to_datetime(df['date'])
        # df.set_index('date', inplace=True) # CNN input function expects 'date' as a column if _dataframe_to_candles is used by rule-based part
        return df

    async def run_test_cnn_patterns():
        cnn_detector = CNNPatterns() # This will now load models based on pattern_configs
        test_symbol_val = 'ETH/USDT' # Renamed variable

        # Ensure candles_by_timeframe_test keys match what detect_patterns_multi_timeframe might look for
        candles_by_timeframe_test = {
            '5m': create_mock_dataframe_for_cnn('5m', 60),
            '1h': create_mock_dataframe_for_cnn('1h', 60),
            # Add other timeframes if your TIME_FRAME_CONFIG in detect_patterns_multi_timeframe expects them
            # For this test, 5m and 1h should be enough to trigger CNN predictions if configured for these TFs
        }

        logger.info(f"--- Test CNNPatterns voor {test_symbol_val} ---")
        # Make sure the detect_patterns_multi_timeframe can handle 'date' not being an index
        # The _dataframe_to_candles method in the provided user code handles it.
        detected_patterns_result = await cnn_detector.detect_patterns_multi_timeframe(candles_by_timeframe_test, test_symbol_val) # Renamed

        print(json.dumps(detected_patterns_result, indent=2, default=str)) # Added default=str for datetime or other non-serializable

        # Verify CNN predictions were attempted and added
        assert 'cnn_predictions' in detected_patterns_result, "Key 'cnn_predictions' not found in results."

        # Check if any score was added. Specific scores depend on model loading and data.
        # Example: Check if bullFlag score for 5m data is present, if model was loaded.
        bullflag_5m_key = "5m_bullFlag_score"
        bearishengulfing_5m_key = "5m_bearishEngulfing_score"

        if cnn_detector.models.get('bullFlag'):
            assert bullflag_5m_key in detected_patterns_result['cnn_predictions'], f"{bullflag_5m_key} not in cnn_predictions."
            logger.info(f"Found {bullflag_5m_key}: {detected_patterns_result['cnn_predictions'][bullflag_5m_key]}")
        else:
            logger.warning("bullFlag model not loaded, skipping assertion for its score.")

        if cnn_detector.models.get('bearishEngulfing'):
            assert bearishengulfing_5m_key in detected_patterns_result['cnn_predictions'], f"{bearishengulfing_5m_key} not in cnn_predictions."
            logger.info(f"Found {bearishengulfing_5m_key}: {detected_patterns_result['cnn_predictions'][bearishengulfing_5m_key]}")
        else:
            logger.warning("bearishEngulfing model not loaded, skipping assertion for its score.")

        # Test specific pattern prediction directly
        if cnn_detector.models.get('bullFlag'): # Check if model is loaded
            logger.info(f"--- Test CNN model prediction for bullFlag on 5m data ---")
            # Ensure the dataframe passed has the 'date' column if _dataframe_to_candles is involved indirectly or if not needed as index
            bull_flag_score_val = await cnn_detector.predict_pattern_score(candles_by_timeframe_test['5m'], 'bullFlag') # Renamed
            logger.info(f"Predicted bullFlag score (5m): {bull_flag_score_val:.4f}")
            assert 0.0 <= bull_flag_score_val <= 1.0, "bullFlag score out of range."
        else:
            logger.warning("bullFlag model not loaded, skipping direct predict_pattern_score test for it.")

    asyncio.run(run_test_cnn_patterns())
