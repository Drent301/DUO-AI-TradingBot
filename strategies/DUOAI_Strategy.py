# strategies/DUOAI_Strategy.py
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import numpy as np

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy.interface import IStrategy # type: ignore
# No more talib import
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import Exchange
from freqtrade.persistence import Trade
from typing import Tuple # For type hinting _calculate_macd_pandas

# Importeer je eigen AI-modules
import os # Added for path operations
import json # Added for json.dump
from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.cnn_patterns import CNNPatterns
from core.reflectie_lus import ReflectieLus
from core.ai_activation_engine import AIActivationEngine # Added import
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.entry_decider import EntryDecider
from core.exit_optimizer import ExitOptimizer
from core.strategy_manager import StrategyManager
from core.interval_selector import IntervalSelector
from core.params_manager import ParamsManager
from core.cooldown_tracker import CooldownTracker # Ensure this is present

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DUOAI_Strategy(IStrategy):
    """
    De hoofdstrategieklasse voor de DUO-AI Trading Bot.
    Deze strategie combineert Freqtrade met onze eigen AI-modules voor
    zelflerende besluitvorming, reflectie en optimalisatie.
    """

    # Freqtrade Hyperopt/Strategy parameters (initial defaults)
    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.02,
        "120": 0.01
    }
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    timeframe = '5m'
    startup_candle_count = 100

    # Define informative timeframes to fetch data for AI modules
    # These are required by `cnn_patterns` and `prompt_builder` for multi-timeframe context.
    # Updated to include all pairs from config.json's pair_whitelist and relevant timeframes.
    informative_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'] # These are the timeframes to merge into base DF

    # Maak instanties van je AI-modules
    prompt_builder: PromptBuilder = PromptBuilder()
    gpt_reflector: GPTReflector = GPTReflector()
    grok_reflector: GrokReflector = GrokReflector()
    cnn_patterns: CNNPatterns = CNNPatterns()
    reflectie_lus: ReflectieLus = ReflectieLus()
    ai_activation_engine: AIActivationEngine # Added instance variable type hint
    bias_reflector: BiasReflector = BiasReflector()
    confidence_engine: ConfidenceEngine = ConfidenceEngine()
    entry_decider: EntryDecider = EntryDecider()
    exit_optimizer: ExitOptimizer = ExitOptimizer()
    strategy_manager: StrategyManager = StrategyManager()
    interval_selector: IntervalSelector = IntervalSelector()
    params_manager: ParamsManager = ParamsManager()
    cooldown_tracker: CooldownTracker = CooldownTracker() # Ensure this is present


    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the DUOAI_Strategy.
        Sets up AI components, loads initial parameters, and defines paths for logging.

        Args:
            config: The configuration dictionary provided by Freqtrade.
        """
        super().__init__(config)

        # Retrieve pair_whitelist from config and store it
        self.config_pair_whitelist = config.get('pair_whitelist', [])

        # Generate informative_pairs based on config_pair_whitelist and informative_timeframes
        self.informative_pairs = [
            (pair, tf)
            for pair in self.config_pair_whitelist
            for tf in DUOAI_Strategy.informative_timeframes
        ]

        # Load initial parameters using a default pair from the whitelist if available
        # This ensures that even before populate_indicators, some learned params are set.
        # self.dp is not available here, so we can't fetch pair-specific data yet.
        # _load_and_apply_learned_parameters will be called again in populate_indicators.
        initial_pair_for_params = self.config_pair_whitelist[0] if self.config_pair_whitelist else 'default'
        self._load_and_apply_learned_parameters(initial_pair_for_params)

        # Initialize AIActivationEngine
        self.ai_activation_engine = AIActivationEngine(reflectie_lus_instance=self.reflectie_lus)

        # Define pattern performance log file path and ensure the 'logs' directory exists.
        # This file is used to log contributing patterns for later performance analysis.
        self.PATTERN_PERFORMANCE_LOG_FILE = os.path.join(
            self.config['user_data_dir'], 'logs', 'pattern_performance_log.json'
        )
        os.makedirs(os.path.dirname(self.PATTERN_PERFORMANCE_LOG_FILE), exist_ok=True)
        logger.info(f"Pattern performance log will be saved to: {self.PATTERN_PERFORMANCE_LOG_FILE}")

        logger.info(f"DUOAI_Strategy geïnitialiseerd.")

    # --- Custom Indicator Calculation Methods (Pandas based) ---
    def _calculate_rsi_pandas(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using Pandas."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)

        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Replace inf values with 100 (where avg_loss is 0)
        rsi = rsi.replace([float('inf'), -float('inf')], 100.0)
        # Fill initial NaNs with 50 (neutral)
        rsi = rsi.fillna(50.0)
        return rsi

    def _calculate_macd_pandas(self, series: pd.Series, fast_p: int = 12, slow_p: int = 26, signal_p: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram using Pandas."""
        ema_fast = series.ewm(span=fast_p, adjust=False, min_periods=fast_p).mean()
        ema_slow = series.ewm(span=slow_p, adjust=False, min_periods=slow_p).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_p, adjust=False, min_periods=signal_p).mean()
        macd_hist = macd_line - signal_line

        return macd_line, signal_line, macd_hist

    def _get_all_relevant_candles_for_ai(self, pair: str) -> Dict[str, pd.DataFrame]:
        """
        Haalt alle relevante candles (basistimeframe + informatives) op voor AI-modules.
        """
        candles_by_timeframe: Dict[str, pd.DataFrame] = {}

        # Base timeframe
        base_df = self.dp.get_pair_dataframe(pair, self.timeframe)
        if not base_df.empty:
            candles_by_timeframe[self.timeframe] = base_df.copy()
        else:
            logger.warning(f"Geen basis dataframe voor {pair} op {self.timeframe}. AI-modules krijgen mogelijk incomplete data.")
            return {} # Return empty if base DF is missing, critical for AI.

        # Informative timeframes
        # Fetches separate DataFrames for each informative timeframe defined in `self.informative_timeframes`.
        # This is crucial for AI modules that need to analyze each timeframe independently.
        for tf in self.informative_timeframes: # self.informative_timeframes = ['1h', '4h', '1d']
            if tf == self.timeframe: # Skip base timeframe as it's already added
                continue
            try:
                # Check if this specific pair and timeframe combination is in `self.informative_pairs`
                # This is an implicit check now, as `self.dp.get_pair_dataframe` will only have data
                # for pairs/timeframes Freqtrade is configured to fetch (which `informative_pairs` controls).
                informative_df = self.dp.get_pair_dataframe(pair, tf) # pair is the current pair being processed
                if not informative_df.empty:
                    candles_by_timeframe[tf] = informative_df.copy()
                else:
                    # This is expected if the pair is not in `informative_pairs` for this `tf`, or no data from exchange
                    logger.debug(f"Informative dataframe for {pair} on {tf} is empty. Skipping.")
            except Exception as e:
                logger.warning(f"Kon informative dataframe voor {pair} op {tf} niet ophalen via self.dp: {e}")

        return candles_by_timeframe

    def _load_and_apply_learned_parameters(self, pair: str) -> None:
        """
        Laadt de laatst bekende geleerde parameters van de `params_manager`
        en past ze toe op de Freqtrade strategie parameters.
        Deze methode kan ook worden aangeroepen om periodiek te updaten.
        """
        # Parameters like ROI, stoploss are typically strategy-wide, not per-pair,
        # but params_manager might store them under a general key or strategy_id.
        # Assuming they are stored under the strategy_id (self.name).
        strategy_params = self.params_manager.get_param("strategies", strategy_id=self.name)

        if strategy_params and isinstance(strategy_params, dict):
            self.minimal_roi = strategy_params.get("minimal_roi", self.minimal_roi)
            self.stoploss = strategy_params.get("stoploss", self.stoploss)
            self.trailing_stop = strategy_params.get("trailing_stop", self.trailing_stop) # Added trailing_stop
            self.trailing_stop_positive = strategy_params.get("trailing_stop_positive", self.trailing_stop_positive)
            self.trailing_stop_positive_offset = strategy_params.get("trailing_stop_positive_offset", self.trailing_stop_positive_offset)
            self.trailing_only_offset_is_reached = strategy_params.get("trailing_only_offset_is_reached", self.trailing_only_offset_is_reached) # Added

            logger.debug(f"Strategie parameters voor {self.name} (context pair: {pair}) bijgewerkt: {strategy_params}")
        else:
            logger.debug(f"Geen geleerde parameters gevonden voor {self.name} in params_manager. Gebruik standaardstrategieparameters.")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Voeg indicatoren toe aan de dataframe en merge informative pairs.
        """
        self._load_and_apply_learned_parameters(metadata['pair']) # Load latest params

        # --- Calculate Base Timeframe Indicators (Pandas / qtpylib) ---
        dataframe['rsi'] = self._calculate_rsi_pandas(dataframe['close'], period=14)

        macd_line, signal_line, macd_hist = self._calculate_macd_pandas(dataframe['close'])
        dataframe['macd'] = macd_line
        dataframe['macdsignal'] = signal_line
        dataframe['macdhist'] = macd_hist

        # Bollinger Bands using qtpylib (uses 'close' by default if no typical price series is given)
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # EMAs using qtpylib or direct Pandas
        dataframe['ema_10'] = qtpylib.ema(dataframe, period=10) # qtpylib.ema uses 'close' column by default
        dataframe['ema_25'] = qtpylib.ema(dataframe, period=25)
        dataframe['ema_50'] = qtpylib.ema(dataframe, period=50)
        # Alternatively, using Pandas directly:
        # dataframe['ema_10'] = dataframe['close'].ewm(span=10, adjust=False, min_periods=10).mean()
        # dataframe['ema_25'] = dataframe['close'].ewm(span=25, adjust=False, min_periods=25).mean()
        # dataframe['ema_50'] = dataframe['close'].ewm(span=50, adjust=False, min_periods=50).mean()

        dataframe['volume_mean_20'] = dataframe['volume'].rolling(window=20).mean() # Numpy based, should be fine

        # --- Candlestick Patterns for Base Timeframe (using pandas_ta) ---
        # Freqtrade dataframes usually have lowercase columns already.
        if not hasattr(dataframe, 'ta') or dataframe.ta is None:
            logger.error("Pandas_ta accessor not found on base dataframe. Skipping candlestick patterns.")
        else:
            candlestick_patterns_map = {
                "cdl_DOJI": "doji",
                "cdl_ENGULFING": "engulfing",
                "cdl_HAMMER": "hammer",
                "cdl_INVERTEDHAMMER": "invertedhammer",
                "cdl_MORNINGSTAR": "morningstar",
                "cdl_EVENINGSTAR": "eveningstar",
                "cdl_SHOOTINGSTAR": "shootingstar",
                "cdl_3WHITESOLDIERS": "3whitesoldiers", # pandas-ta uses cdl_3whitesoldiers
                "cdl_3BLACKCROWS": "3blackcrows",     # pandas-ta uses cdl_3blackcrows
                "cdl_HARAMI": "harami"
            }
            for col_name, pta_func_suffix in candlestick_patterns_map.items():
                try:
                    # Ensure OHLC columns exist for base dataframe
                    if not all(c in dataframe.columns for c in ['open', 'high', 'low', 'close']):
                        logger.warning(f"Base dataframe missing one or more OHLC columns. Cannot calculate {col_name}.")
                        dataframe[col_name] = 0
                        continue

                    indicator_func = getattr(dataframe.ta, f"cdl_{pta_func_suffix}")
                    result = indicator_func()
                    if result is not None:
                        dataframe[col_name] = result.fillna(0)
                    else:
                        logger.warning(f"Candlestick pattern {pta_func_suffix} for base timeframe returned None. Filling {col_name} with 0.")
                        dataframe[col_name] = 0
                except AttributeError:
                    logger.warning(f"Pandas_ta candlestick pattern cdl_{pta_func_suffix} not found for base timeframe. Filling {col_name} with 0.")
                    dataframe[col_name] = 0
                except Exception as e_cdl_base:
                    logger.warning(f"Error calculating {col_name} for base timeframe: {e_cdl_base}. Filling {col_name} with 0.")
                    dataframe[col_name] = 0
        # --- End Candlestick Patterns for Base Timeframe ---

        # --- Merge informative timeframes into the main dataframe ---
        # This uses `self.informative_pairs` to tell Freqtrade which (pair, timeframe)
        # combinations it needs to have data for. Freqtrade's DataProvider handles fetching these.
        # `merge_informative_pair` then merges *only the data for the current pair* (metadata['pair'])
        # from the specified informative timeframes into the base dataframe.
        for pair_to_merge, informative_tf_to_merge in self.informative_pairs:
            if pair_to_merge == metadata['pair'] and informative_tf_to_merge != self.timeframe:
                informative_df = self.dp.get_pair_dataframe(pair_to_merge, informative_tf_to_merge)
                if not informative_df.empty:
                    dataframe = merge_informative_pair(dataframe, informative_df, self.timeframe, informative_tf_to_merge, fqtrade=True)
                else:
                    logger.debug(f"Informative dataframe for {pair_to_merge} on {informative_tf_to_merge} is empty. Skipping merge for {metadata['pair']}.")

        # --- Calculate indicators for merged informative timeframes ---
        for info_tf in self.informative_timeframes: # e.g., '1h', '4h', '1d'
            if info_tf == self.timeframe: # Should not happen if logic is correct, but good check
                continue

            prefix = f"{info_tf}" # Standard Freqtrade prefix for merged columns

            # Check if the primary column (e.g., '{prefix}_close') exists before proceeding
            if f'{prefix}_close' not in dataframe.columns:
                logger.debug(f"Skipping indicator calculation for {info_tf} as '{prefix}_close' is not in dataframe.")
                continue

            logger.debug(f"Calculating indicators for informative timeframe {info_tf} using Pandas/qtpylib.")

            # RSI for informative timeframe
            dataframe[f'{prefix}_rsi'] = self._calculate_rsi_pandas(dataframe[f'{prefix}_close'], period=14)

            # MACD for informative timeframe
            inf_macd_line, inf_signal_line, inf_macd_hist = self._calculate_macd_pandas(dataframe[f'{prefix}_close'])
            dataframe[f'{prefix}_macd'] = inf_macd_line
            dataframe[f'{prefix}_macdsignal'] = inf_signal_line
            dataframe[f'{prefix}_macdhist'] = inf_macd_hist

            # Bollinger Bands for informative timeframe
            # qtpylib.bollinger_bands defaults to using the 'close' column of the passed series.
            # Here, we pass the specific '{prefix}_close' series.
            bollinger_inf = qtpylib.bollinger_bands(dataframe[f'{prefix}_close'], window=20, stds=2)
            dataframe[f'{prefix}_bb_lowerband'] = bollinger_inf['lower']
            dataframe[f'{prefix}_bb_middleband'] = bollinger_inf['mid']
            dataframe[f'{prefix}_bb_upperband'] = bollinger_inf['upper']

            # EMAs for informative timeframe
            # qtpylib.ema expects a full dataframe and a column name, or uses 'close' by default.
            # To use it with prefixed columns, we can create a temporary Series or use Pandas' ewm directly.
            # Using Pandas ewm for clarity with prefixed columns:
            dataframe[f'{prefix}_ema_10'] = dataframe[f'{prefix}_close'].ewm(span=10, adjust=False, min_periods=10).mean()
            dataframe[f'{prefix}_ema_25'] = dataframe[f'{prefix}_close'].ewm(span=25, adjust=False, min_periods=25).mean()
            dataframe[f'{prefix}_ema_50'] = dataframe[f'{prefix}_close'].ewm(span=50, adjust=False, min_periods=50).mean()
            # Alternatively, using qtpylib.ema by renaming the column temporarily (less clean):
            # temp_series_for_ema = dataframe[f'{prefix}_close'].rename('close')
            # dataframe[f'{prefix}_ema_10'] = qtpylib.ema(temp_series_for_ema.to_frame(), period=10) # Needs a DataFrame

            # Volume Mean (e.g., volume_mean_20) - Numpy based
            if f'{prefix}_volume' in dataframe.columns:
                dataframe[f'{prefix}_volume_mean_20'] = dataframe[f'{prefix}_volume'].rolling(window=20).mean()
            else:
                logger.debug(f"Skipping volume mean calculation for {info_tf} as '{prefix}_volume' is not in dataframe.")

            # --- Candlestick Patterns for Informative Timeframe: {prefix} ---
            # candlestick_patterns_map is defined in the base timeframe section above and can be reused
            if 'candlestick_patterns_map' not in locals() and 'candlestick_patterns_map' not in globals():
                 # Redefine if somehow not in scope (e.g. if base patterns part was skipped)
                 candlestick_patterns_map = {
                    "cdl_DOJI": "doji", "cdl_ENGULFING": "engulfing", "cdl_HAMMER": "hammer",
                    "cdl_INVERTEDHAMMER": "invertedhammer", "cdl_MORNINGSTAR": "morningstar",
                    "cdl_EVENINGSTAR": "eveningstar", "cdl_SHOOTINGSTAR": "shootingstar",
                    "cdl_3WHITESOLDIERS": "3whitesoldiers", "cdl_3BLACKCROWS": "3blackcrows",
                    "cdl_HARAMI": "harami"
                 }

            if not hasattr(dataframe, 'ta') or dataframe.ta is None:
                logger.error(f"Pandas_ta accessor not found on dataframe (for informative {prefix}). Skipping candlestick patterns.")
            else:
                required_info_ohlc = [f'{prefix}_open', f'{prefix}_high', f'{prefix}_low', f'{prefix}_close']
                if all(c in dataframe.columns for c in required_info_ohlc):
                    for col_name_suffix, pta_func_suffix in candlestick_patterns_map.items():
                        target_col_name = f"{prefix}_{col_name_suffix}"
                        try:
                            indicator_func = getattr(dataframe.ta, f"cdl_{pta_func_suffix}")
                            result = indicator_func(
                                open=dataframe[f'{prefix}_open'],
                                high=dataframe[f'{prefix}_high'],
                                low=dataframe[f'{prefix}_low'],
                                close=dataframe[f'{prefix}_close']
                            )
                            if result is not None:
                                dataframe[target_col_name] = result.fillna(0)
                            else:
                                logger.warning(f"Candlestick pattern {pta_func_suffix} for informative {prefix} returned None. Filling {target_col_name} with 0.")
                                dataframe[target_col_name] = 0
                        except AttributeError:
                            logger.warning(f"Pandas_ta candlestick pattern cdl_{pta_func_suffix} not found for informative {prefix}. Filling {target_col_name} with 0.")
                            dataframe[target_col_name] = 0
                        except Exception as e_cdl_info:
                            logger.warning(f"Error calculating {target_col_name} for informative {prefix}: {e_cdl_info}. Filling {target_col_name} with 0.")
                            dataframe[target_col_name] = 0
                else:
                    logger.warning(f"Missing one or more OHLC columns for prefix {prefix}. Skipping candlestick patterns for this informative timeframe.")
                    for col_name_suffix, _ in candlestick_patterns_map.items():
                        dataframe[f"{prefix}_{col_name_suffix}"] = 0
            # --- End Candlestick Patterns for Informative Timeframe ---

        # Haal bias en confidence op (van AI-modules) en voeg toe aan dataframe
        current_bias = self.bias_reflector.get_bias_score(metadata['pair'], self.name)
        current_confidence = self.confidence_engine.get_confidence_score(metadata['pair'], self.name)
        dataframe['ai_bias'] = current_bias
        dataframe['ai_confidence'] = current_confidence

        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            leverage: float, default_stake_amount: float, **kwargs) -> float:
        """
        Pas de inzet per trade aan op basis van AI-confidence (maxTradeRiskPct).
        """
        max_per_trade_pct_learned = self.confidence_engine.confidence_memory.get(pair, {}).get(self.name, {}).get('max_per_trade_pct', 0.1)
        # Get global maxTradeRiskPct from params_manager (not strategy specific)
        global_max_trade_risk = self.params_manager.get_param("maxTradeRiskPct", strategy_id=None)

        effective_max_trade_risk = min(max_per_trade_pct_learned, global_max_trade_risk)

        total_capital = self.wallets.get_leverage_capital(self.stake_currency)
        adjusted_stake_amount = total_capital * effective_max_trade_risk

        if self.stake_amount != "unlimited": # Ensure it respects Freqtrade's stake_amount if not unlimited
            adjusted_stake_amount = min(adjusted_stake_amount, float(self.stake_amount))

        min_stake_from_config = self.config.get('exchange', {}).get('min_stake_amount', 10.0) # Get min_stake from config if possible
        adjusted_stake_amount = max(adjusted_stake_amount, min_stake_from_config)


        logger.info(f"Aangepaste stake amount voor {pair}: {adjusted_stake_amount:.2f} {self.stake_currency} (gebaseerd op effectieve MaxPerTradePct: {effective_max_trade_risk:.2%}).")
        return adjusted_stake_amount

    async def custom_entry(self, pair: str, current_time: datetime,
                           dataframe: pd.DataFrame, **kwargs) -> Optional[float]:
        """
        AI-gestuurde entry-beslissing.
        Retrieves an entry decision from the EntryDecider. If an entry is signaled,
        the patterns contributing to this decision are logged to `pattern_performance_log.json`
        for later analysis by the PatternPerformanceAnalyzer.

        Args:
            pair: The trading pair (e.g., "ETH/USDT").
            current_time: The datetime of the current candle.
            dataframe: The Freqtrade DataFrame for the current pair and base timeframe.
            **kwargs: Additional keyword arguments passed by Freqtrade.

        Returns:
            Optional[float]: 1.0 to signal an entry, or None to decline entry.
        """
        candles_by_timeframe_for_ai = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_ai: # Check if dictionary is empty (e.g., base_df was empty)
            logger.warning(f"Geen voldoende dataframes voor AI-entrybesluit voor {pair}. Entry geweigerd.")
            return None

        learned_bias = self.bias_reflector.get_bias_score(pair, self.name)
        learned_confidence = self.confidence_engine.get_confidence_score(pair, self.name)
        # Get strategy-specific entryConvictionThreshold
        entry_conviction_threshold = self.params_manager.get_param("entryConvictionThreshold", strategy_id=self.name)
        # Cooldown duration is global, not strategy-specific in this setup
        # cooldown_duration_seconds = self.params_manager.get_param("cooldownDurationSeconds", strategy_id=None)

        # AI-specifieke cooldown check using CooldownTracker
        if self.cooldown_tracker.is_cooldown_active(pair, self.name):
            cooldown_info = self.cooldown_tracker._cooldown_state.get(pair, {}).get(self.name, {})
            cooldown_reason = cooldown_info.get('reason', 'unknown')
            cooldown_end_time_str = cooldown_info.get('end_time', 'N/A')
            logger.info(f"Entry geweigerd voor {pair} door AI-specifieke cooldown (reden: {cooldown_reason}, eindigt: {cooldown_end_time_str}).")
            return None


        entry_decision = await self.entry_decider.should_enter(
            dataframe=dataframe, # Base timeframe DataFrame from Freqtrade
            symbol=pair,
            current_strategy_id=self.name,
            trade_context={"current_price": dataframe['close'].iloc[-1], "timeframe": self.timeframe, "candles_by_timeframe": candles_by_timeframe_for_ai},
            learned_bias=learned_bias,
            learned_confidence=learned_confidence,
            entry_conviction_threshold=entry_conviction_threshold
        )

        if entry_decision.get('enter'):
            logger.info(f"Entry toegestaan voor {pair}. Reden: {entry_decision['reason']}. Confidence: {entry_decision['confidence']:.2f}")

            # Log contributing patterns if entry is signaled
            contributing_patterns = entry_decision.get('contributing_patterns', [])
            log_data = {
                "pair": pair,
                "entry_timestamp": current_time.isoformat(), # Freqtrade provides current_time as datetime
                "contributing_patterns": contributing_patterns,
                "decision_details": entry_decision # Optionally log the full decision for more context
            }
            try:
                # Append the log entry as a new JSON line to the performance log file
                with open(self.PATTERN_PERFORMANCE_LOG_FILE, 'a', encoding='utf-8') as f:
                    json.dump(log_data, f)
                    f.write('\n')
            except Exception as e:
                logger.error(f"Fout bij schrijven naar pattern_performance_log.json: {e}")

            return 1.0 # Signal Freqtrade to enter

        logger.info(f"Entry geweigerd voor {pair}. Reden: {entry_decision['reason']}")
        return None

    async def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                          dataframe: pd.DataFrame, **kwargs) -> Optional[float]:
        """
        AI-gestuurde exit-beslissing en dynamische SL/TP aanpassing.
        """
        candles_by_timeframe_for_ai = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_ai:
            logger.warning(f"Geen voldoende dataframes voor AI-exitbesluit voor {pair}. Exit overgeslagen.")
            return None # Skip exit evaluation if data is incomplete

        learned_bias = self.bias_reflector.get_bias_score(pair, self.name)
        learned_confidence = self.confidence_engine.get_confidence_score(pair, self.name)
        # Get strategy-specific exitConvictionDropTrigger
        exit_conviction_drop_trigger = self.params_manager.get_param("exitConvictionDropTrigger", strategy_id=self.name)

        exit_decision = await self.exit_optimizer.should_exit(
            dataframe=dataframe, # Base timeframe DataFrame
            trade=trade.to_json(),
            symbol=pair,
            current_strategy_id=self.name,
            candles_by_timeframe=candles_by_timeframe_for_ai,
            learned_bias=learned_bias,
            learned_confidence=learned_confidence,
            exit_conviction_drop_trigger=exit_conviction_drop_trigger
        )

        if exit_decision['exit']:
            logger.info(f"Exit getriggerd voor {pair}. Reden: {exit_decision['reason']}. Confidence: {exit_decision['confidence']:.2f}")
            # Potentially activate cooldown upon AI-driven exit
            await self.cooldown_tracker.activate_cooldown(pair, self.name, reason=f"ai_exit_{exit_decision['reason']}")
            return dataframe['close'].iloc[-1] # Signal Freqtrade to exit at current price

        # Dynamische Trailing Stop Loss Optimalisatie (via AI)
        sl_optimization_result = await self.exit_optimizer.optimize_trailing_stop_loss(
            dataframe=dataframe,
            trade=trade.to_json(),
            symbol=pair,
            current_strategy_id=self.name,
            candles_by_timeframe=candles_by_timeframe_for_ai,
            learned_bias=learned_bias,
            learned_confidence=learned_confidence
        )

        if sl_optimization_result:
            self.stoploss = sl_optimization_result.get("stoploss", self.stoploss)
            self.trailing_stop_positive_offset = sl_optimization_result.get("trailing_stop_positive_offset", self.trailing_stop_positive_offset)
            self.trailing_stop_positive = sl_optimization_result.get("trailing_stop_positive", self.trailing_stop_positive)
            # Optionally, update trailing_stop and trailing_only_offset_is_reached if AI provides them
            # self.trailing_stop = sl_optimization_result.get("trailing_stop", self.trailing_stop)
            # self.trailing_only_offset_is_reached = sl_optimization_result.get("trailing_only_offset_is_reached", self.trailing_only_offset_is_reached)

            asyncio.create_task(self.params_manager.update_strategy_roi_sl_params(
                strategy_id=self.name,
                new_roi=self.minimal_roi, # ROI is not typically optimized here, but included for completeness
                new_stoploss=self.stoploss,
                new_trailing_stop_positive=self.trailing_stop_positive,
                new_trailing_stop_positive_offset=self.trailing_stop_positive_offset
                # Pass other params if they are also being updated:
                # new_trailing_stop=self.trailing_stop,
                # new_trailing_only_offset_is_reached=self.trailing_only_offset_is_reached
            ))
            logger.info(f"TSL parameters bijgewerkt voor {pair}: Stoploss={self.stoploss}, TSL_Offset={self.trailing_stop_positive_offset}, TSL_Trigger={self.trailing_stop_positive}")


        return None # Geen onmiddellijke exit via AI, Freqtrade's own SL/ROI/TSL will apply

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Extra checks voor de AI-consensus en confidence vlak voor de trade.
        """
        current_confidence = self.confidence_engine.get_confidence_score(pair, self.name)
        current_bias = self.bias_reflector.get_bias_score(pair, self.name)
        entry_conviction_threshold = self.params_manager.get_param("entryConvictionThreshold", strategy_id=self.name)

        if current_confidence < entry_conviction_threshold or current_bias < 0.5: # Example bias threshold
            logger.warning(f"Trade entry voor {pair} geweigerd in confirm_trade_entry. Lage confidence ({current_confidence:.2f}) of bias ({current_bias:.2f}) of onder drempel ({entry_conviction_threshold:.2f}).")
            return False

        logger.info(f"Trade entry voor {pair} bevestigd in confirm_trade_entry. Conf: {current_confidence:.2f}, Bias: {current_bias:.2f}.")
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Extra checks voor de AI-consensus en confidence vlak voor de exit.
        """
        logger.info(f"Trade exit voor {pair} bevestigd in confirm_trade_exit.")
        return True

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Freqtrade's entry trend populatie.
        De AI-gestuurde logica is in `custom_entry`. Hier zetten we een placeholder
        om `custom_entry` te triggeren.
        """
        dataframe.loc[
            (dataframe['volume'] > 0), # Always True if there's any volume
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Freqtrade's exit trend populatie.
        De AI-gestuurde logica is in `custom_exit`. Hier zetten we een placeholder
        om `custom_exit` te triggeren.
        """
        dataframe.loc[
            (dataframe['volume'] > 0), # Always True if there's any volume
            'exit_long'] = 1
        return dataframe

    def process_stopped_trade(self, pair: str, trade: Trade, order: Dict[str, Any], **kwargs) -> None:
        """
        Deze methode wordt aangeroepen nadat een trade is gesloten.
        Ideaal voor het triggeren van AI-reflectie en het bijwerken van leerbare variabelen.
        Ook voor het activeren van een cooldown na een verlies.
        """
        profit_loss_pct = trade.calc_profit_ratio()

        trade_data = trade.to_json()
        trade_data['profit_pct'] = profit_loss_pct
        trade_data['exit_rate'] = order.get('price')
        trade_data['exit_type'] = order.get('ft_pair_exit_reason', 'unknown') # Freqtrade specific exit reason

        logger.info(f"Trade gesloten voor {pair} (ID: {trade.id}). Resultaat: {profit_loss_pct:.2%}. Exit reden: {trade_data['exit_type']}. Trigger AI reflectie en leerloops.")

        # Activeer cooldown als de trade verliesgevend was
        if profit_loss_pct < 0:
            loss_cooldown_reason = f"loss_{trade_data['exit_type']}_{profit_loss_pct:.2%}"
            asyncio.create_task(self.cooldown_tracker.activate_cooldown(pair, self.name, reason=loss_cooldown_reason))

        # Trigger de AI-reflectie
        candles_by_timeframe_for_reflect = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_reflect:
            logger.warning(f"Geen voldoende dataframes voor post-trade reflectie voor {pair}. Reflectie overgeslagen.")
            # We don't return here, as performance update should still happen.
        else: # Only run reflection if data is available
            asyncio.create_task(
                self.ai_activation_engine.activate_ai(
                    trigger_type='trade_closed',
                    token=pair,
                    candles_by_timeframe=candles_by_timeframe_for_reflect,
                    strategy_id=self.name,
                    trade_context=trade_data,
                    mode=self.config.get('runmode', 'live'),
                    bias_reflector_instance=self.bias_reflector,
                    confidence_engine_instance=self.confidence_engine
                )
            )

        # Update strategy performance in strategy_manager
        current_perf = self.strategy_manager.get_strategy_performance(self.name)
        new_trade_count = current_perf.get('tradeCount', 0) + 1
        # Correct calculation for total profit based on average profit and new trade
        old_total_profit = current_perf.get('avgProfit', 0.0) * current_perf.get('tradeCount', 0)
        new_total_profit_value = old_total_profit + profit_loss_pct

        new_win_rate = (current_perf.get('winRate', 0.0) * current_perf.get('tradeCount', 0) + (1 if profit_loss_pct > 0 else 0)) / new_trade_count if new_trade_count > 0 else 0.0
        new_avg_profit = new_total_profit_value / new_trade_count if new_trade_count > 0 else 0.0

        asyncio.create_task(self.strategy_manager.update_strategy_performance(
            strategy_id=self.name,
            new_performance={
                "winRate": new_win_rate,
                "avgProfit": new_avg_profit,
                "tradeCount": new_trade_count,
                # "totalProfit": new_total_profit_value # Optionally track total profit sum
            }
        ))
