# strategies/DUOAI_Strategy.py
from freqtrade.strategy import IStrategy, merge_informative_pair, stoploss_to_trend
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List # Added List
import asyncio # Added for asyncio.create_task

# Importeer je eigen AI-modules
# Assuming 'core' is in PYTHONPATH or accessible relative to strategies
import sys
import os
# Add project root to sys.path if core modules are not found directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.cnn_patterns import CNNPatterns
from core.reflectie_lus import ReflectieLus
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.entry_decider import EntryDecider
from core.exit_optimizer import ExitOptimizer

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Freqtrade usually manages logging levels via config

class DUOAI_Strategy(IStrategy):
    """
    De hoofdstrategieklasse voor de DUO-AI Trading Bot.
    """

    minimal_roi = {
        "0": 0.05, "30": 0.03, "60": 0.02, "120": 0.01
    }
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    timeframe = '5m'
    startup_candle_count = 100

    _ai_modules_initialized = False
    # Declare class variables for AI modules for clarity
    prompt_builder: Optional[PromptBuilder] = None
    gpt_reflector: Optional[GPTReflector] = None
    grok_reflector: Optional[GrokReflector] = None
    cnn_patterns: Optional[CNNPatterns] = None
    reflectie_lus: Optional[ReflectieLus] = None
    bias_reflector: Optional[BiasReflector] = None
    confidence_engine: Optional[ConfidenceEngine] = None
    entry_decider: Optional[EntryDecider] = None
    exit_optimizer: Optional[ExitOptimizer] = None
    _candles_by_timeframe_cache: Dict[str, Dict[str, DataFrame]] = {}


    def __init__(self, config: dict):
        super().__init__(config)
        if not DUOAI_Strategy._ai_modules_initialized:
            logger.info("Initializing AI modules for DUOAI_Strategy...")
            try:
                DUOAI_Strategy.prompt_builder = PromptBuilder()
            except Exception as e:
                logger.error(f"Failed to initialize PromptBuilder: {e}")

            DUOAI_Strategy.gpt_reflector = GPTReflector()
            DUOAI_Strategy.grok_reflector = GrokReflector()
            DUOAI_Strategy.cnn_patterns = CNNPatterns()

            try: # ReflectieLus might depend on PromptBuilder
                DUOAI_Strategy.reflectie_lus = ReflectieLus()
            except Exception as e:
                logger.error(f"Failed to initialize ReflectieLus: {e}")

            DUOAI_Strategy.bias_reflector = BiasReflector()

            try:
                DUOAI_Strategy.confidence_engine = ConfidenceEngine()
            except Exception as e:
                logger.error(f"Failed to initialize ConfidenceEngine: {e}")

            # Initialize EntryDecider and ExitOptimizer, checking dependencies
            if DUOAI_Strategy.prompt_builder and DUOAI_Strategy.confidence_engine and                DUOAI_Strategy.gpt_reflector and DUOAI_Strategy.grok_reflector and                DUOAI_Strategy.bias_reflector and DUOAI_Strategy.cnn_patterns:
                try:
                    DUOAI_Strategy.entry_decider = EntryDecider()
                except Exception as e:
                    logger.error(f"Failed to initialize EntryDecider: {e}")
                try:
                    DUOAI_Strategy.exit_optimizer = ExitOptimizer()
                except Exception as e:
                    logger.error(f"Failed to initialize ExitOptimizer: {e}")
            else:
                logger.error("Cannot initialize EntryDecider/ExitOptimizer due to missing dependencies.")

            DUOAI_Strategy._ai_modules_initialized = True
            logger.info("AI modules initialization attempt complete.")

        # Initialize cache if it's not already (for multiple instantiations if that happens)
        if not hasattr(DUOAI_Strategy, '_candles_by_timeframe_cache') or DUOAI_Strategy._candles_by_timeframe_cache is None: # Check for None too
             DUOAI_Strategy._candles_by_timeframe_cache = {}


    def _get_pair_timeframe_cache(self, pair: str) -> Dict[str, DataFrame]:
        if pair not in DUOAI_Strategy._candles_by_timeframe_cache:
            DUOAI_Strategy._candles_by_timeframe_cache[pair] = {}
        return DUOAI_Strategy._candles_by_timeframe_cache[pair]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        current_timeframe_df_cache = self._get_pair_timeframe_cache(pair)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['volume_mean_20'] = dataframe['volume'].rolling(window=20, min_periods=1).mean()

        current_timeframe_df_cache[self.timeframe] = dataframe.copy()

        if DUOAI_Strategy.bias_reflector and DUOAI_Strategy.confidence_engine:
            current_bias = DUOAI_Strategy.bias_reflector.get_bias_score(pair, self.name)
            current_confidence = DUOAI_Strategy.confidence_engine.get_confidence_score(pair, self.name)
            dataframe['ai_bias'] = current_bias
            dataframe['ai_confidence'] = current_confidence
        else:
            dataframe['ai_bias'] = 0.5
            dataframe['ai_confidence'] = 0.5

        return dataframe

    async def custom_entry(self, pair: str, current_time: datetime, dataframe: DataFrame, **kwargs) -> bool:
        if not DUOAI_Strategy.entry_decider:
            logger.warning(f"EntryDecider niet beschikbaar voor {pair}. Entry overgeslagen.")
            return False

        pair_cache = self._get_pair_timeframe_cache(pair)
        pair_cache[self.timeframe] = dataframe.copy() # Update with latest dataframe

        # The current EntryDecider.should_enter expects only the primary dataframe.
        # If it needs more, it should use self.dp or have its signature changed.
        entry_decision = await DUOAI_Strategy.entry_decider.should_enter(
            dataframe=dataframe,
            symbol=pair,
            current_strategy_id=self.name
        )

        if entry_decision and entry_decision.get('enter'):
            logger.info(f"AI Entry GOEDKEURING voor {pair}. Reden: {entry_decision.get('reason')}. AI Conf: {entry_decision.get('confidence',0):.2f}")
            return True

        logger.debug(f"AI Entry GEWEIGERD voor {pair}. Reden: {entry_decision.get('reason', 'onbekend') if entry_decision else 'geen besluit'}")
        return False


    async def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float,
                          current_profit: float, dataframe: DataFrame, **kwargs) -> Optional[str]:
        if not DUOAI_Strategy.exit_optimizer:
            logger.warning(f"ExitOptimizer niet beschikbaar voor {pair}. Exit overgeslagen.")
            return None

        pair_cache = self._get_pair_timeframe_cache(pair)
        pair_cache[self.timeframe] = dataframe.copy()

        exit_decision = await DUOAI_Strategy.exit_optimizer.should_exit(
            dataframe=dataframe,
            trade=trade.to_json(),
            symbol=pair,
            current_strategy_id=self.name
        )

        if exit_decision and exit_decision.get('exit'):
            logger.info(f"AI Exit GOEDKEURING voor {pair}. Reden: {exit_decision.get('reason')}. AI Conf: {exit_decision.get('confidence',0):.2f}")
            return exit_decision.get('reason', 'ai_custom_exit')

        # SL optimization logic might be better in adjust_trade_position or a custom_stoploss function
        # For now, we are not calling it from custom_exit to avoid complexity here.
        # It can be called by the ReflectieLus or a dedicated SL management part.

        return None

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_above(dataframe['rsi'], 30)) &
            (dataframe['macd'] > dataframe['macdsignal']),
            ['enter_long', 'enter_tag']] = (1, 'rsi_macd_cross_entry')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_below(dataframe['rsi'], 70)) &
            (dataframe['macd'] < dataframe['macdsignal']),
            ['exit_long', 'exit_tag']] = (1, 'rsi_macd_cross_exit')
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Any, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:

        if DUOAI_Strategy.reflectie_lus and DUOAI_Strategy.bias_reflector and DUOAI_Strategy.confidence_engine:
            trade_data_for_reflection = trade.to_json()
            profit_pct = trade.calc_profit_ratio(rate)
            trade_data_for_reflection['profit_pct'] = profit_pct # Add profit to context

            logger.info(f"Trade exit bevestigd voor {pair} (reden: {sell_reason}). Voorbereiden AI-reflectie. Winst: {profit_pct:.2%}")

            pair_cache = self._get_pair_timeframe_cache(pair)
            # Ensure we have some data for reflection, even if it's just the current timeframe
            candles_for_reflection = {tf: df.copy() for tf, df in pair_cache.items() if not df.empty}
            if not candles_for_reflection and self.timeframe in pair_cache and not pair_cache[self.timeframe].empty:
                 candles_for_reflection = {self.timeframe: pair_cache[self.timeframe].copy()}
            elif not candles_for_reflection: # Still no data
                 logger.warning(f"Geen candle data beschikbaar in cache voor {pair} voor reflectie.")
                 # Create a dummy if absolutely necessary, but ideally this shouldn't happen
                 # candles_for_reflection = DUOAI_Strategy.reflectie_lus._create_mock_dataframe_for_reflection(pair)


            # Schedule reflection task only if there's data
            if candles_for_reflection and DUOAI_Strategy.reflectie_lus: # Check if reflectie_lus is initialized
                try:
                    current_bias = DUOAI_Strategy.bias_reflector.get_bias_score(pair, self.name)
                    current_confidence = DUOAI_Strategy.confidence_engine.get_confidence_score(pair, self.name)

                    asyncio.create_task(
                        DUOAI_Strategy.reflectie_lus.process_reflection_cycle(
                            symbol=pair,
                            candles_by_timeframe=candles_for_reflection,
                            strategy_id=self.name,
                            trade_context=trade_data_for_reflection,
                            current_bias=current_bias,
                            current_confidence=current_confidence,
                            mode='live_trade_closed'
                        )
                    )
                except Exception as e:
                    logger.error(f"Fout bij starten AI-reflectie taak voor {pair} na exit: {e}")

            # Direct bias/confidence update
            try:
                asyncio.create_task(DUOAI_Strategy.bias_reflector.update_bias(
                    token=pair, strategy_id=self.name,
                    new_ai_bias=DUOAI_Strategy.bias_reflector.get_bias_score(pair, self.name),
                    confidence=DUOAI_Strategy.confidence_engine.get_confidence_score(pair, self.name),
                    trade_result_pct=profit_pct
                ))
                asyncio.create_task(DUOAI_Strategy.confidence_engine.update_confidence(
                    token=pair, strategy_id=self.name,
                    ai_confidence=DUOAI_Strategy.confidence_engine.get_confidence_score(pair, self.name),
                    trade_result_pct=profit_pct
                ))
            except Exception as e:
                 logger.error(f"Fout bij directe bias/confidence update na {pair} exit: {e}")

        return True
