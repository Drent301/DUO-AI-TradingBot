# core/cnn_patterns.py
import logging
import os
import json
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import talib # Voor candlestick patronen die TA-Lib biedt

# PyTorch Imports
import torch
import torch.nn as nn
from core.pre_trainer import SimpleCNN # Import SimpleCNN class

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CNNPatterns:
    """
    Detecteert visuele candlestick- en technische patronen in marktdata.
    Integreert nu een PyTorch CNN model voor patroonherkenning.
    """

    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'models')
    CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_patterns_model.pth')
    SCALER_PARAMS_PATH = os.path.join(MODELS_DIR, 'cnn_scaler_params.json') # Path for scaler parameters
    os.makedirs(MODELS_DIR, exist_ok=True)

    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.scaler_feature_columns: Optional[List[str]] = None
        self.scaler_min_vals: Optional[np.ndarray] = None
        self.scaler_max_vals: Optional[np.ndarray] = None
        self._load_ml_models()
        logger.info("CNNPatterns geïnitialiseerd.")

    def _load_ml_models(self):
        """
        Laadt het getrainde PyTorch SimpleCNN-model.
        """
        """
        Laadt het getrainde PyTorch SimpleCNN-model en de scaler parameters.
        """
        # Load CNN model
        logger.info(f"Poging om PyTorch ML-model te laden van {CNNPatterns.CNN_MODEL_PATH}...")
        if os.path.exists(CNNPatterns.CNN_MODEL_PATH):
            try:
                # Assuming feature_columns are known or derived from scaler_params
                # For now, using a typical number of input_channels. This will be verified by scaler_params.
                # input_channels = 9 # This will be determined by len(self.scaler_feature_columns) later
                num_classes = 2     # bullFlag_label (0 of 1)

                # Defer model instantiation until scaler_params are loaded to know input_channels
                # model = SimpleCNN(input_channels=input_channels, num_classes=num_classes)
                # model._set_num_classes_for_fc_init(num_classes)
                # model.load_state_dict(torch.load(CNNPatterns.CNN_MODEL_PATH, map_location=torch.device('cpu')))
                # model.eval()
                # self.loaded_models['cnn_pattern_recognizer'] = model
                # logger.info(f"PyTorch model 'cnn_pattern_recognizer' succesvol geladen van {CNNPatterns.CNN_MODEL_PATH}.")
                self.model_state_dict = torch.load(CNNPatterns.CNN_MODEL_PATH, map_location=torch.device('cpu'))
                logger.info(f"PyTorch model state_dict succesvol geladen van {CNNPatterns.CNN_MODEL_PATH}.")

            except Exception as e:
                logger.error(f"Fout bij het laden van PyTorch model van {CNNPatterns.CNN_MODEL_PATH}: {e}")
                self.model_state_dict = None
        else:
            logger.warning(f"PyTorch model niet gevonden op {CNNPatterns.CNN_MODEL_PATH}.")
            self.model_state_dict = None

        # Load scaler parameters
        logger.info(f"Poging om CNN scaler parameters te laden van {CNNPatterns.SCALER_PARAMS_PATH}...")
        if os.path.exists(CNNPatterns.SCALER_PARAMS_PATH):
            try:
                with open(CNNPatterns.SCALER_PARAMS_PATH, 'r') as f:
                    scaler_params = json.load(f)
                self.scaler_feature_columns = scaler_params['feature_columns']
                self.scaler_min_vals = np.array(scaler_params['min_vals'])
                self.scaler_max_vals = np.array(scaler_params['max_vals'])
                logger.info(f"CNN scaler parameters succesvol geladen. Features: {self.scaler_feature_columns}")

                # Now instantiate the model if state_dict was loaded
                if self.model_state_dict and self.scaler_feature_columns:
                    input_channels = len(self.scaler_feature_columns)
                    num_classes = 2 # Assuming 2 classes
                    model = SimpleCNN(input_channels=input_channels, num_classes=num_classes)
                    model._set_num_classes_for_fc_init(num_classes)
                    model.load_state_dict(self.model_state_dict)
                    model.eval()
                    self.loaded_models['cnn_pattern_recognizer'] = model
                    logger.info(f"PyTorch model 'cnn_pattern_recognizer' geïnstantieerd en state_dict toegepast.")
                elif not self.model_state_dict:
                    logger.warning("Scaler parameters geladen, maar model state_dict niet. CNN model niet volledig operationeel.")


            except Exception as e:
                logger.error(f"Fout bij het laden of verwerken van scaler parameters van {CNNPatterns.SCALER_PARAMS_PATH}: {e}")
                self.scaler_feature_columns = None
                self.scaler_min_vals = None
                self.scaler_max_vals = None
        else:
            logger.warning(f"CNN scaler parameters niet gevonden op {CNNPatterns.SCALER_PARAMS_PATH}. CNN-voorspellingen zullen niet mogelijk zijn.")
            self.scaler_feature_columns = None
            self.scaler_min_vals = None
            self.scaler_max_vals = None

    def _predict_pattern_score(self, model_name: str, input_candles_df: pd.DataFrame) -> float:
        """
        Voert inferentie uit met een geladen PyTorch ML-model om een patroonscore te voorspellen.
        Gebruikt nu geladen scaler parameters.
        """
        model = self.loaded_models.get(model_name)
        if not model:
            logger.warning(f"Model '{model_name}' niet gevonden of niet correct geladen in self.loaded_models.")
            return 0.0

        if not self.scaler_feature_columns or self.scaler_min_vals is None or self.scaler_max_vals is None:
            logger.warning(f"Scaler parameters (feature_columns, min_vals, max_vals) niet geladen voor model '{model_name}'. Kan geen voorspelling doen.")
            return 0.0

        if input_candles_df is None or input_candles_df.empty:
            logger.debug(f"Input DataFrame voor '{model_name}' is leeg of None.")
            return 0.0

        sequence_length = 30  # Moet overeenkomen met training (of uit params_manager halen)

        if len(input_candles_df) < sequence_length:
            logger.debug(f"Niet genoeg data in input_candles_df ({len(input_candles_df)} rijen) voor sequence_length {sequence_length} voor model '{model_name}'.")
            return 0.0

        # Ensure all required features are present
        if not all(col in input_candles_df.columns for col in self.scaler_feature_columns):
            missing_cols = [col for col in self.scaler_feature_columns if col not in input_candles_df.columns]
            logger.warning(f"Ontbrekende feature kolommen in input_candles_df voor model '{model_name}': {missing_cols}. Kan geen voorspelling doen.")
            return 0.0

        # Neem de laatste 'sequence_length' candles en selecteer/order features
        df_sequence = input_candles_df.iloc[-sequence_length:].copy()
        sequence_data_pd = df_sequence[self.scaler_feature_columns] # Select and order features

        # Normalisatie met geladen scaler parameters
        # Ensure sequence_data_pd columns are in the same order as self.scaler_min_vals/max_vals
        # This is implicitly handled if self.scaler_feature_columns was used for selection and ordering

        # Convert to numpy array for scaling, if not already
        # sequence_data_np = sequence_data_pd.values # This is already done by accessing .values later

        range_vals = self.scaler_max_vals - self.scaler_min_vals
        range_vals[range_vals == 0] = 1 # Replicate training logic for constant features

        # Apply scaling: (data - min) / range
        # Ensure scaler_min_vals and range_vals are broadcastable to sequence_data_pd
        # sequence_data_pd is (sequence_length, num_features)
        # self.scaler_min_vals is (num_features,)
        normalized_sequence_np = (sequence_data_pd.values - self.scaler_min_vals) / range_vals

        # Tensor Conversion: (batch_size=1, num_features/input_channels, sequence_length)
        # input_tensor expects (num_input_channels, sequence_length) before unsqueeze
        # num_input_channels is num_features
        input_tensor = torch.tensor(normalized_sequence_np, dtype=torch.float32).permute(1, 0).unsqueeze(0)

        if input_tensor.shape[1] != len(self.scaler_feature_columns):
            logger.error(f"Mismatch in tensor input channels ({input_tensor.shape[1]}) and expected from scaler ({len(self.scaler_feature_columns)}).")
            return 0.0

        logger.debug(f"Tensor voor model '{model_name}' shape: {input_tensor.shape}")

        try:
            with torch.no_grad():
                output_logits = model(input_tensor)

            # Pas softmax toe om waarschijnlijkheden te krijgen
            probabilities = torch.softmax(output_logits, dim=1)

            # Waarschijnlijkheid van de positieve klasse (index 1) en negatieve klasse (index 0)
            positive_class_probability = probabilities[0, 1].item() # bullFlag_score
            negative_class_probability = probabilities[0, 0].item() # no_bullFlag_score (or other interpretation)

            logger.debug(f"Model '{model_name}' voorspelde logits: {output_logits.numpy()}")
            logger.info(f"Model '{model_name}' probabilities: bullFlag_score={positive_class_probability:.4f}, no_bullFlag_score={negative_class_probability:.4f}")

            return {
                'bullFlag_score': positive_class_probability,
                'no_bullFlag_score': negative_class_probability
            }
        except Exception as e:
            logger.error(f"Fout tijdens inferentie met model '{model_name}': {e}")
            return {'bullFlag_score': 0.0, 'no_bullFlag_score': 0.0} # Return dict even on error

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

            ha_high = max(c['high'], ha_open, ha_close)
            ha_low = min(c['low'], ha_open, ha_close)
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

        # --- ML Model Inference ---
        ml_model_name = 'cnn_pattern_recognizer'
        if ml_model_name in self.loaded_models:
            # Kies een geschikte DataFrame voor voorspelling. '15m' of '1h' zijn vaak goede kandidaten.
            # De keuze kan afhangen van de aard van het getrainde model.
            df_for_prediction = candles_by_timeframe.get('15m')

            if df_for_prediction is not None and not df_for_prediction.empty:
                prediction_timeframe = '15m' # Explicitly define the TF used for this prediction
                logger.info(f"Uitvoeren van ML voorspelling met '{ml_model_name}' op {symbol} ({prediction_timeframe} data).")

                cnn_scores_dict = self._predict_pattern_score(ml_model_name, df_for_prediction)

                if cnn_scores_dict:
                    for score_key, score_value in cnn_scores_dict.items():
                        # Prefix with timeframe, e.g., "15m_bullFlag_score"
                        prefixed_key = f"{prediction_timeframe}_{score_key}"
                        all_patterns["cnn_predictions"][prefixed_key] = score_value
                        logger.info(f"ML Model '{ml_model_name}' ({prediction_timeframe}) voorspelde score voor '{prefixed_key}': {score_value:.4f}")
                else:
                    logger.warning(f"ML model '{ml_model_name}' gaf geen scores terug voor {symbol} ({prediction_timeframe}).")

            else:
                logger.info(f"Geen geschikte data (15m) gevonden voor ML voorspelling ({ml_model_name}) voor {symbol}. Overslaan.")
        else:
            logger.info(f"ML model '{ml_model_name}' niet geladen. Overslaan ML-gebaseerde patroon detectie.")

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
    # import os # os is already imported at the top level
    import json # Toegevoegd voor json.dumps in test
    import dotenv
    from pathlib import Path # Import Path
    # Corrected path for .env when running this script directly
    # dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    dotenv_path = Path(__file__).resolve().parent.parent / '.env'
    dotenv.load_dotenv(dotenv_path)


    # Mock Freqtrade DataFrames voor verschillende timeframes
    def create_mock_dataframe(timeframe: str, num_candles: int = 100) -> pd.DataFrame:
        data = []
        # Gebruik UTC voor consistentie
        now = datetime.utcnow() # pd.Timestamp.utcnow().to_pydatetime()
        interval_seconds_map = {
            '1m': 60, '5m': 300, '15m': 900, '1h': 3600,
            '4h': 14400, '12h': 43200, '1d': 86400, '1w': 604800
        }
        interval_seconds = interval_seconds_map.get(timeframe, 300) # Default to 5m if not found


        for i in range(num_candles):
            # Genereer timestamps in het verleden
            date = now - timedelta(seconds=(num_candles - 1 - i) * interval_seconds)
            open_ = 100 + i * 0.1 + np.random.rand() * 2
            close_ = open_ + (np.random.rand() - 0.5) * 5 # Kan positief of negatief zijn
            high_ = max(open_, close_) + np.random.rand() * 2
            low_ = min(open_, close_) - np.random.rand() * 2
            volume = 1000 + np.random.rand() * 500

            # Voeg hier mock indicatorwaarden toe die Freqtrade zou berekenen
            rsi = 50 + (np.random.rand() - 0.5) * 30
            macd_val = (np.random.rand() - 0.5) * 0.1 # Kleinere waarden voor MACD
            macdsignal_val = macd_val * (0.8 + np.random.rand() * 0.2) # Signaal lijn dichtbij MACD
            macdhist_val = macd_val - macdsignal_val

            # Basis voor Bollinger Bands
            sma_period = 20
            std_dev_multiplier = 2
            # Ensure there's enough data for SMA calculation for current candle 'i'
            # We need 'sma_period' previous data points including the current one to form the list for SMA.
            # So, we need at least 'sma_period -1' historical points before 'i'.
            # The data list 'data' is being built up. For the i-th candle (0-indexed),
            # we need data[i-sma_period+1] up to data[i].
            # This means i must be at least sma_period - 1.

            current_candle_data_for_sma = []
            if i >= sma_period -1:
                 # Extract close prices from 'data' list for SMA calculation
                 # 'data' contains lists like [date, open_, high_, low_, close_, volume, ...]
                 # The close price is at index 4.
                current_candle_data_for_sma = [d[4] for d in data[i-sma_period+1:i+1]]


            if current_candle_data_for_sma and len(current_candle_data_for_sma) == sma_period :
                sma = np.mean(current_candle_data_for_sma)
                std_dev = np.std(current_candle_data_for_sma)
                bb_middle = sma
                bb_upper = sma + std_dev_multiplier * std_dev
                bb_lower = sma - std_dev_multiplier * std_dev
            else: # Fallback if not enough data for SMA
                bb_middle = open_
                bb_upper = high_
                bb_lower = low_


            data.append([date, open_, high_, low_, close_, volume, rsi, macd_val, macdsignal_val, macdhist_val, bb_upper, bb_middle, bb_lower])

        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_upperband', 'bb_middleband', 'bb_lowerband'])
        # Converteer 'date' kolom naar datetime objecten en zet als index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        df.attrs['timeframe'] = timeframe # Voeg timeframe toe als attribuut
        return df

    async def run_test_cnn_patterns():
        # Create a dummy cnn_scaler_params.json for testing
        dummy_scaler_params = {
            "feature_columns": ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist'],
            "min_vals": [90, 95, 85, 90, 1000, 20, -0.5, -0.4, -0.1],
            "max_vals": [110, 115, 105, 110, 5000, 80, 0.5, 0.4, 0.1]
        }
        # Ensure MODELS_DIR exists for the dummy file
        os.makedirs(CNNPatterns.MODELS_DIR, exist_ok=True)
        with open(CNNPatterns.SCALER_PARAMS_PATH, 'w') as f:
            json.dump(dummy_scaler_params, f, indent=4)
        logger.info(f"Dummy scaler params created at {CNNPatterns.SCALER_PARAMS_PATH} for testing.")

        # Create a dummy model state_dict for testing _load_ml_models
        # This needs a valid SimpleCNN state_dict. For simplicity, we'll skip creating a full dummy model state_dict.
        # Instead, we'll rely on the logs from _load_ml_models to indicate if scaler loading works.
        # If CNN_MODEL_PATH does not exist, the model won't load, but scaler loading should still be testable.
        # To test model instantiation with scaler, a dummy model file would be needed.
        # For now, let's assume it's okay if the model itself doesn't load, but scaler params do.

        cnn_detector = CNNPatterns() # This will call _load_ml_models

        if cnn_detector.scaler_feature_columns:
            logger.info(f"SUCCESS: Scaler feature columns loaded: {cnn_detector.scaler_feature_columns}")
            logger.info(f"SUCCESS: Scaler min_vals loaded (shape): {cnn_detector.scaler_min_vals.shape}")
            logger.info(f"SUCCESS: Scaler max_vals loaded (shape): {cnn_detector.scaler_max_vals.shape}")
        else:
            logger.error("FAILURE: Scaler parameters were not loaded during CNNPatterns initialization.")


        test_symbol = 'BTC/USDT'
        num_features = len(dummy_scaler_params["feature_columns"])

        # Creëer mock dataframes voor multi-timeframe test
        # Ensure mock dataframes include all columns listed in dummy_scaler_params["feature_columns"]
        candles_by_timeframe = {
            '1m': create_mock_dataframe('1m', 60), # num_features will be matched by create_mock_dataframe
            '5m': create_mock_dataframe('5m', 60),
            '15m': create_mock_dataframe('15m', 60),
            '1h': create_mock_dataframe('1h', 60),
            '4h': create_mock_dataframe('4h', 60),
            '1d': create_mock_dataframe('1d', 60),
        }

        print(f"\n--- Test CNNPatterns voor {test_symbol} ---")
        # logger.info("Starting multi-timeframe pattern detection test...")

        # Check if the model was actually loaded before trying to use it
        if 'cnn_pattern_recognizer' in cnn_detector.loaded_models:
            logger.info("CNN model 'cnn_pattern_recognizer' is loaded. Proceeding with full detection.")
            detected_patterns = await cnn_detector.detect_patterns_multi_timeframe(candles_by_timeframe, test_symbol)
            print("\n--- Detected Patterns (including CNN predictions) ---")
            print(json.dumps(detected_patterns, indent=2, default=str))

            # Verify cnn_predictions structure
            if "cnn_predictions" in detected_patterns:
                logger.info("SUCCESS: 'cnn_predictions' key found in output.")
                if "15m_bullFlag_score" in detected_patterns["cnn_predictions"] and \
                   "15m_no_bullFlag_score" in detected_patterns["cnn_predictions"]:
                    logger.info("SUCCESS: Timeframe-prefixed CNN scores (e.g., '15m_bullFlag_score') found.")
                else:
                    logger.error("FAILURE: Expected timeframe-prefixed CNN scores not found in 'cnn_predictions'.")
            else:
                logger.error("FAILURE: 'cnn_predictions' key NOT found in output.")

        else:
            logger.warning("CNN model 'cnn_pattern_recognizer' not loaded. Skipping detect_patterns_multi_timeframe which relies on it.")
            # Create a dummy detected_patterns structure for the rest of the test if model isn't loaded
            detected_patterns = {"patterns": {}, "cnn_predictions": {}, "context": {}}


    # Ensure logger is configured for __main__ to see output if using logger.debug in detect_patterns_multi_timeframe
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    # print(json.dumps(detected_patterns, indent=2, default=str)) # Moved up

        print("\n--- Specifieke patroon tests op 5m (Rule-based) ---")
        mock_5m_df = candles_by_timeframe['5m']
        # Reset index to make 'date' a column for _dataframe_to_candles
        mock_5m_candles_list = cnn_detector._dataframe_to_candles(mock_5m_df.reset_index())
        if mock_5m_candles_list: # Alleen als de lijst niet leeg is
            print(f"Bull Flag (5m): {cnn_detector.detect_bull_flag(mock_5m_candles_list)}")
            print(f"Breakout (5m): {cnn_detector.detect_breakout(mock_5m_candles_list)}")
            print(f"EMA Cross (5m): {cnn_detector.detect_ema_cross(mock_5m_candles_list)}")
            print(f"Engulfing (5m, custom): {cnn_detector._detect_engulfing(mock_5m_candles_list)}")
            print(f"Bollinger Squeeze (5m): {cnn_detector.detect_bollinger_squeeze(mock_5m_candles_list)}")
        else:
            print("Kon 5m candle lijst niet genereren voor specifieke tests.")


        # Test TA-Lib patronen (vereist DataFrame input)
        if not mock_5m_df.empty:
             talib_patterns_5m = cnn_detector.detect_candlestick_patterns(mock_5m_df)
             print(f"Doji (5m, TA-Lib): {talib_patterns_5m.get('CDLDOJI', False)}")
        else:
            print("5m DataFrame is leeg, overslaan TA-Lib tests.")

        # Clean up dummy scaler file
        if os.path.exists(CNNPatterns.SCALER_PARAMS_PATH):
            os.remove(CNNPatterns.SCALER_PARAMS_PATH)
            logger.info(f"Dummy scaler params file {CNNPatterns.SCALER_PARAMS_PATH} removed.")

    asyncio.run(run_test_cnn_patterns())
