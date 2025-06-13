# core/ai_activation_engine.py
import logging
import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np # For mock data generation in test
import dotenv # Added for __main__

# Importeer AI-modules
from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.cnn_patterns import CNNPatterns
from core.reflectie_lus import ReflectieLus
# ReflectieLus, BiasReflector, ConfidenceEngine, StrategyManager
# will be passed via dependency injection to activate_ai to avoid circular imports.

from core.grok_sentiment_fetcher import GrokSentimentFetcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Padconfiguratie
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
REFLECTIE_LOG_FILE = os.path.join(MEMORY_DIR, 'reflectie-logboek.json') # Same log file as ReflectieLus

os.makedirs(MEMORY_DIR, exist_ok=True)

class AIActivationEngine:
    """
    Trigger-engine voor AI bij events: entry, exit, reflection, CNN-triggers.
    Vertaald en geoptimaliseerd van aiActivationEngine.js.
    """

    def __init__(self, reflectie_lus_instance: ReflectieLus):
        try:
            self.prompt_builder = PromptBuilder()
        except Exception as e:
            logger.error(f"Failed to initialize PromptBuilder in AIActivationEngine: {e}")
            self.prompt_builder = None # Fallback

        self.reflectie_lus_instance = reflectie_lus_instance
        if self.reflectie_lus_instance is None:
            logger.error("ReflectieLus instance not provided to AIActivationEngine. AI activation will not work correctly.")
            # Optionally raise an error: raise ValueError("ReflectieLus instance is required")

        self.gpt_reflector = GPTReflector()
        self.grok_reflector = GrokReflector()
        self.cnn_patterns_detector = CNNPatterns()

        try:
            self.grok_sentiment_fetcher = GrokSentimentFetcher()
        except ValueError as e:
            logger.error(f"Failed to initialize GrokSentimentFetcher: {e}. Sentiment analysis will be disabled.")
            self.grok_sentiment_fetcher = None

        self.last_reflection_timestamps: Dict[str, float] = {} # Per token
CNN_DETECTION_THRESHOLD = 0.7 # Class constant for CNN detection threshold
        logger.info("AIActivationEngine geïnitialiseerd.")

    def _get_time_since_last_reflection(self, token: str) -> float:
        return (datetime.now().timestamp() * 1000) - self.last_reflection_timestamps.get(token, 0)

    def _update_last_reflection_timestamp(self, token: str):
        self.last_reflection_timestamps[token] = datetime.now().timestamp() * 1000

    async def _should_trigger_ai(self, trigger_data: Dict[str, Any], symbol: str, mode: str = 'live') -> bool:
        """
        Bepaalt of de AI-reflectie moet worden getriggerd op basis van verschillende factoren.
        """
        score = 0

        if self.grok_sentiment_fetcher:
            try:
                sentiment_items = await self.grok_sentiment_fetcher.fetch_live_search_data(symbol)
                if sentiment_items:
                    logger.info(f"Sentiment items found for {symbol}: {len(sentiment_items)}. Incrementing trigger score.")
                    score += 1 # Increment score by 1 if sentiment items are found
                else:
                    logger.info(f"No sentiment items found for {symbol}.")
            except Exception as e:
                logger.error(f"Error fetching sentiment for {symbol}: {e}")
        else:
            logger.warning("GrokSentimentFetcher not available, skipping sentiment check in _should_trigger_ai.")

        pattern_score = trigger_data.get('patternScore', 0.0)
        volume_spike = trigger_data.get('volumeSpike', False)
        learned_confidence = trigger_data.get('learned_confidence', 0.5)
        time_since_last = trigger_data.get('time_since_last_reflection', 0)
        profit_metric = trigger_data.get('profit_metric', 0.0)

        if pattern_score > 0.7: score += 2
        if volume_spike: score += 1
        if learned_confidence < 0.4: score += 2
        elif learned_confidence < 0.6: score +=1

        if time_since_last > 60 * 60 * 1000: score += 1
        if profit_metric < -0.01: score += 2
        elif profit_metric < 0: score +=1

        trigger_threshold = self.get_dynamic_trigger_threshold(learned_confidence, mode)

        if mode == 'pretrain' or mode == 'backtest_reflection':
            logger.debug(f"AI trigger FORCED for {mode} mode.")
            return True

        triggered = score >= trigger_threshold
        logger.debug(f"AI Trigger Check: Score={score}, Threshold={trigger_threshold}, Triggered={triggered}. Data: {trigger_data}")
        return triggered

    def get_dynamic_trigger_threshold(self, learned_confidence: float, mode: str) -> int:
        """ Dynamische drempelwaarde voor AI activatie. """
        if mode == 'highly_explorative': return 1
        base_threshold = 3
        if learned_confidence < 0.5:
            base_threshold = 2
        return base_threshold

    async def _log_reflection_entry(self, entry: Dict[str, Any]):
        """ Slaat een reflectie-entry op in het logboek. """
        logs = []
        try:
            if os.path.exists(REFLECTIE_LOG_FILE) and os.path.getsize(REFLECTIE_LOG_FILE) > 0:
                with open(REFLECTIE_LOG_FILE, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                    if not isinstance(logs, list): logs = []
        except json.JSONDecodeError:
            logger.warning(f"Reflectie logboek {REFLECTIE_LOG_FILE} is leeg of corrupt. Start met lege lijst.")
        except FileNotFoundError:
             logger.warning(f"Reflectie logboek {REFLECTIE_LOG_FILE} niet gevonden. Start met lege lijst.")

        logs.append(entry)
        try:
            # Use await asyncio.to_thread for blocking file I/O
            def save_log_sync():
                with open(REFLECTIE_LOG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, indent=2)
            await asyncio.to_thread(save_log_sync)
        except IOError as e:
            logger.error(f"Fout bij opslaan reflectielogboek: {e}")


    async def activate_ai(
        self,
        trigger_type: str,
        token: str,
        candles_by_timeframe: Dict[str, pd.DataFrame],
        strategy_id: str,
        trade_context: Optional[Dict[str, Any]] = None,
        mode: str = 'live',
        bias_reflector_instance: Optional[Any] = None,
        confidence_engine_instance: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Activeert AI-reflectie op basis van een trigger-type en context.
        """
        logger.info(f"[AI-ACTIVATION] Trigger: {trigger_type} voor {token} (Strategie: {strategy_id}) in Mode: {mode}...")

        if self.prompt_builder is None:
            logger.error("PromptBuilder niet geïnitialiseerd in AIActivationEngine. Kan AI niet activeren.")
            return None

        current_bias = bias_reflector_instance.get_bias_score(token, strategy_id) if bias_reflector_instance else 0.5
        learned_confidence = confidence_engine_instance.get_confidence_score(token, strategy_id) if confidence_engine_instance else 0.5

        pattern_data = await self.cnn_patterns_detector.detect_patterns_multi_timeframe(candles_by_timeframe, token)

        # These are the patterns specifically analyzed by CNN models
        cnn_specific_pattern_keys = self.cnn_patterns_detector.get_all_detectable_pattern_keys()
        total_possible_patterns = len(cnn_specific_pattern_keys)

        pattern_score_val = 0.0
        if total_possible_patterns > 0:
            cnn_detected_count = 0
            if pattern_data and pattern_data.get('cnn_predictions'):
                # Iterate through the patterns that have dedicated CNN models
                for cnn_pattern_key in cnn_specific_pattern_keys:
                    # Check if this CNN pattern was detected with high confidence on any timeframe
                    pattern_detected_on_any_tf = False
                    for prediction_key, score in pattern_data['cnn_predictions'].items():
                        # prediction_key might be like "5m_bullFlag_score" or "1h_bearishEngulfing_score"
                        # We check if the cnn_pattern_key (e.g., "bullFlag") is part of the prediction_key
                        # and ends with "_score"
                        if f"_{cnn_pattern_key}_score" in prediction_key and prediction_key.endswith("_score"):
                            if score >= self.CNN_DETECTION_THRESHOLD: # Use class constant
                                pattern_detected_on_any_tf = True
                                break  # Found for this cnn_pattern_key, move to next
                    if pattern_detected_on_any_tf:
                        cnn_detected_count += 1

            pattern_score_val = cnn_detected_count / total_possible_patterns
        else:
            # This case handles when there are no CNN-specific patterns defined.
            # Could also log a warning if this is unexpected.
            logger.info("No CNN-specific patterns defined for detection, pattern_score_val is 0.")

        trigger_data = {
            "patternScore": pattern_score_val,
            "volumeSpike": pattern_data.get('context', {}).get('volume_spike', False) if pattern_data else False,
            "learned_confidence": learned_confidence,
            "time_since_last_reflection": self._get_time_since_last_reflection(token),
            "profit_metric": trade_context.get('profit_pct', 0.0) if trade_context else 0.0
        }

        if not await self._should_trigger_ai(trigger_data, token, mode):
            logger.info(f"[AI-ACTIVATION] AI niet getriggerd voor {token} ({trigger_type}).")
            return None

        if self.reflectie_lus_instance is None:
            logger.error("ReflectieLus instance is not available in AIActivationEngine. Cannot process reflection.")
            return None

        # Prepare trade_context for ReflectieLus
        # It already contains trade_context passed to activate_ai.
        # We need to add trigger_type and pattern_data.
        extended_trade_context = {
            **(trade_context or {}),
            "trigger_type": trigger_type, # Pass the original trigger_type
            # pattern_data is already computed and available in this scope
        }

        logger.info(f"[AI-ACTIVATION] Delegating to ReflectieLus for {token} ({trigger_type}).")

        # Determine prompt_type based on trigger_type
        prompt_type_mapping = {
            'entry_signal': 'entry_analysis',
            'trade_closed': 'post_trade_analysis',
            'cnn_pattern_detected': 'pattern_analysis'
        }
        prompt_type = prompt_type_mapping.get(trigger_type, 'general_analysis')

        logger.info(f"[AI-ACTIVATION] Determined prompt_type: {prompt_type} for trigger_type: {trigger_type}")

        # The prompt_type for ReflectieLus will be handled in Step 3 of the main plan.
        # For now, we assume ReflectieLus.process_reflection_cycle will take pattern_data.
        # We will pass 'comprehensive_analysis' as prompt_type, and pattern_data.
        # This requires ReflectieLus.process_reflection_cycle to be updated in the next step.
        reflection_result = await self.reflectie_lus_instance.process_reflection_cycle(
            symbol=token,
            candles_by_timeframe=candles_by_timeframe,
            strategy_id=strategy_id,
            trade_context=extended_trade_context, # Contains original trade_context, trigger_type
            current_bias=current_bias, # Calculated in activate_ai
            current_confidence=learned_confidence, # Calculated in activate_ai
            mode=mode,
            # These are new parameters to be added to ReflectieLus.process_reflection_cycle in plan step 3
            prompt_type=prompt_type, # Use the determined prompt_type
            pattern_data=pattern_data
        )

        # The original activate_ai returned reflectie_log_entry.
        # process_reflection_cycle returns a dictionary which might be different.
        # We return what ReflectieLus returns.
        return reflection_result

if __name__ == "__main__":
    # Adjust __main__ to pass ReflectieLus instance
    from core.reflectie_lus import ReflectieLus # Ensure import if not already at top for __main__
    # Mock or initialize ReflectieLus as needed for testing
    # For example, a simple mock if ReflectieLus has complex dependencies:
    class MockReflectieLus:
        async def process_reflection_cycle(self, **kwargs):
            logger.info(f"MockReflectieLus.process_reflection_cycle called with {kwargs.get('symbol')}, {kwargs.get('prompt_type')}")
            # Return a structure similar to what ReflectieLus would return if needed for testing continuity
            return {"reflection_summary": "Mock reflection processed", "details": kwargs}

    mock_reflectie_lus_instance = MockReflectieLus()
    # Pass it to the engine
    # engine = AIActivationEngine() # OLD
    # engine = AIActivationEngine(reflectie_lus_instance=mock_reflectie_lus_instance) # NEW - this will be adjusted in the main block below.

    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    def create_mock_dataframe_for_ai_activation(timeframe: str = '5m', num_candles: int = 100) -> pd.DataFrame:
        data = []
        now = datetime.utcnow()
        interval_seconds_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
        interval_seconds = interval_seconds_map.get(timeframe, 300)
        for i in range(num_candles):
            date = now - timedelta(seconds=(num_candles - 1 - i) * interval_seconds)
            open_ = 100 + i * 0.01 + np.random.randn() * 0.1
            close_ = open_ + np.random.randn() * 0.5
            high_ = max(open_, close_) + np.random.rand() * 0.2
            low_ = min(open_, close_) - np.random.rand() * 0.2
            volume = 1000 + np.random.rand() * 2000
            data.append([date, open_, high_, low_, close_, volume])
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.attrs['timeframe'] = timeframe
        df.attrs['pair'] = 'ETH/USDT'
        return df

    async def run_test_ai_activation_engine():
        # Moved reflectie_lus instance creation inside the async test function
        # to ensure it's created within the same async context if needed,
        # or just for neatness.
        class MockReflectieLusForTest: # Renamed to avoid conflict if MockReflectieLus is defined globally
            def __init__(self):
                self.last_prompt_type = None
                self.last_pattern_data = None
                self.last_symbol = None
                self.last_trade_context = None

            async def process_reflection_cycle(self, **kwargs):
                self.last_prompt_type = kwargs.get('prompt_type')
                self.last_pattern_data = kwargs.get('pattern_data')
                self.last_symbol = kwargs.get('symbol')
                self.last_trade_context = kwargs.get('trade_context')
                logger.info(f"MockReflectieLusForTest.process_reflection_cycle called for {self.last_symbol} with prompt_type {self.last_prompt_type}")
                # Simulate returning a reflection log entry structure
                return {
                    "timestamp": datetime.now().isoformat(),
                    "token": self.last_symbol,
                    "strategyId": kwargs.get('strategy_id'),
                    "prompt_type": self.last_prompt_type,
                    "summary": "Mocked reflection cycle completed.",
                    "mode": kwargs.get('mode'),
                    "pattern_data_used": self.last_pattern_data is not None,
                    "received_trade_context": self.last_trade_context
                }
        mock_reflectie_lus = MockReflectieLusForTest()
        engine = AIActivationEngine(reflectie_lus_instance=mock_reflectie_lus)
        test_token = "ETH/USDT"
        test_strategy_id = "DUOAI_Strategy"

        mock_candles_by_timeframe = {
            '5m': create_mock_dataframe_for_ai_activation('5m', 60),
            '1h': create_mock_dataframe_for_ai_activation('1h', 60)
        }

        class MockBiasReflector:
            def get_bias_score(self, token, strategy_id): return 0.65
        class MockConfidenceEngine:
            def get_confidence_score(self, token, strategy_id): return 0.75

        class MockCNNPatterns(CNNPatterns): # Extend the actual class for testing
             def get_all_detectable_pattern_keys(self) -> list:
                # This method needs to be actually implemented in the real CNNPatterns class
                # For now, this mock only matters if the logic in _should_trigger_ai uses it.
                # The current pattern_score_val calculation is simplified.
                return ["bullFlag", "bearFlag", "breakout", "rsiDivergence", "CDLDOJI", "doubleTop", "headAndShoulders"]
        engine.cnn_patterns_detector = MockCNNPatterns()

        mock_bias_reflector = MockBiasReflector()
        mock_confidence_engine = MockConfidenceEngine()

        # PromptBuilder mock might still be needed if CNNPatterns or other parts of AIActivationEngine use it.
        # If PromptBuilder is solely for ReflectieLus now, this specific mock for AIActivationEngine might be removable.
        # However, CNNPatterns is initialized in AIActivationEngine and might use PromptBuilder.
        # Let's assume it's still potentially needed for other internal mechanics or future features.
        if engine.prompt_builder is None: # This check is good practice.
            class MockPromptBuilder: # Ensure this mock is sufficient for any remaining uses.
                async def generate_prompt_with_data(self, **kwargs):
                    # This mock might not be hit if PromptBuilder is only used by the removed logic.
                    logger.info(f"MockPromptBuilder.generate_prompt_with_data called for {kwargs.get('symbol')}")
                    return f"Fallback mock prompt for {kwargs.get('symbol')} - {kwargs.get('prompt_type')}"
            engine.prompt_builder = MockPromptBuilder()


        print("\n--- Test AIActivationEngine ---")

        # Test 'entry_signal'
        print("\nActiveren AI voor een 'entry_signal' trigger...")
        entry_trade_context = {"signal_strength": 0.85, "some_other_entry_info": "test_value"}
        entry_reflection = await engine.activate_ai(
            trigger_type='entry_signal',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context=entry_trade_context,
            mode='dry_run',
            bias_reflector_instance=mock_bias_reflector,
            confidence_engine_instance=mock_confidence_engine
        )
        if entry_reflection:
            assert mock_reflectie_lus.last_prompt_type == 'entry_analysis', \
                f"Incorrect prompt_type for entry_signal: {mock_reflectie_lus.last_prompt_type}"
            assert mock_reflectie_lus.last_pattern_data is not None, "Pattern data not passed for entry_signal"
            assert mock_reflectie_lus.last_trade_context.get("signal_strength") == 0.85, "Trade context not passed correctly for entry_signal"
            print(f"Resultaat Entry Signaal Reflectie: OK (prompt_type='{mock_reflectie_lus.last_prompt_type}')")
            # print(json.dumps(entry_reflection, indent=2, default=str)) # Optional: print full result
        else:
            print("Entry Signaal Reflectie niet getriggerd / geen resultaat. Test FAILED.")
            assert False, "Entry signal reflection was not triggered."

        # Test 'trade_closed'
        print("\nActiveren AI voor een 'trade_closed' trigger...")
        closed_trade_context = {"entry_price": 2500, "exit_price": 2450, "profit_pct": -0.02, "trade_id": "t123"}
        exit_reflection = await engine.activate_ai(
            trigger_type='trade_closed',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context=closed_trade_context,
            mode='live',
            bias_reflector_instance=mock_bias_reflector,
            confidence_engine_instance=mock_confidence_engine
        )
        if exit_reflection:
            assert mock_reflectie_lus.last_prompt_type == 'post_trade_analysis', \
                f"Incorrect prompt_type for trade_closed: {mock_reflectie_lus.last_prompt_type}"
            assert mock_reflectie_lus.last_pattern_data is not None, "Pattern data not passed for trade_closed"
            assert mock_reflectie_lus.last_trade_context.get("trade_id") == "t123", "Trade context not passed correctly for trade_closed"
            print(f"Resultaat Trade Closed Reflectie: OK (prompt_type='{mock_reflectie_lus.last_prompt_type}')")
            # print(json.dumps(exit_reflection, indent=2, default=str)) # Optional: print full result
        else:
            print("Trade Closed Reflectie niet getriggerd / geen resultaat. Test FAILED.")
            assert False, "Trade closed reflection was not triggered."

        # Test 'cnn_pattern_detected'
        print("\nActiveren AI voor een 'cnn_pattern_detected' trigger...")
        cnn_context = {"pattern_name": "bull_flag_5m", "confidence": 0.92}
        cnn_reflection = await engine.activate_ai(
            trigger_type='cnn_pattern_detected',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context=cnn_context,
            mode='live', # Or any mode that allows triggering
            bias_reflector_instance=mock_bias_reflector,
            confidence_engine_instance=mock_confidence_engine
        )
        if cnn_reflection:
            assert mock_reflectie_lus.last_prompt_type == 'pattern_analysis', \
                f"Incorrect prompt_type for cnn_pattern_detected: {mock_reflectie_lus.last_prompt_type}"
            assert mock_reflectie_lus.last_pattern_data is not None, "Pattern data not passed for cnn_pattern_detected"
            assert mock_reflectie_lus.last_trade_context.get("pattern_name") == "bull_flag_5m", "Trade context not passed correctly for cnn_pattern_detected"
            print(f"Resultaat CNN Pattern Reflectie: OK (prompt_type='{mock_reflectie_lus.last_prompt_type}')")
        else:
            print("CNN Pattern Reflectie niet getriggerd / geen resultaat. Test FAILED.")
            # This might fail if _should_trigger_ai is strict; adjust trigger_data or mock _should_trigger_ai if needed for this test
            # For now, we assume it can trigger.
            assert False, "CNN Pattern reflection was not triggered."

        # Test unknown trigger type
        print("\nActiveren AI voor een onbekende trigger type ('unknown_signal')...")
        unknown_context = {"detail": "some_random_event"}
        unknown_reflection = await engine.activate_ai(
            trigger_type='unknown_signal', # An unknown trigger
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context=unknown_context,
            mode='live',
            bias_reflector_instance=mock_bias_reflector,
            confidence_engine_instance=mock_confidence_engine
        )
        if unknown_reflection:
            assert mock_reflectie_lus.last_prompt_type == 'general_analysis', \
                f"Incorrect prompt_type for unknown_signal: {mock_reflectie_lus.last_prompt_type}"
            assert mock_reflectie_lus.last_pattern_data is not None, "Pattern data not passed for unknown_signal"
            assert mock_reflectie_lus.last_trade_context.get("detail") == "some_random_event", "Trade context not passed correctly for unknown_signal"
            print(f"Resultaat Onbekend Signaal Reflectie: OK (prompt_type='{mock_reflectie_lus.last_prompt_type}')")
        else:
            print("Onbekend Signaal Reflectie niet getriggerd / geen resultaat. Test FAILED.")
            # This might fail if _should_trigger_ai is strict.
            assert False, "Unknown signal reflection was not triggered."

        # Logging is now handled by ReflectieLus, so REFLECTIE_LOG_FILE check here might be redundant
        # or should point to wherever ReflectieLus logs, if different.
        # For now, let's assume ReflectieLus uses the same logging mechanism or path.
        print(f"Check logs for ReflectieLus processing details (potentially {REFLECTIE_LOG_FILE} if used by ReflectieLus).")

        # Call the new test function
        await test_should_trigger_ai_logic(engine)
        # Call the pattern score calculation test function
        await test_pattern_score_val_calculation()

    # Add to imports in __main__ or at the top of the file if used more broadly
    from unittest.mock import patch, AsyncMock

    async def test_pattern_score_val_calculation():
        logger.info("\n--- Test pattern_score_val Calculation ---")

        class MockReflectieLusForTestPatternScore:
            async def process_reflection_cycle(self, **kwargs):
                # This mock is minimal as activate_ai should not reach this point in this test
                logger.info(f"MockReflectieLusForTestPatternScore.process_reflection_cycle called with {kwargs.get('symbol')}")
                return {"summary": "Mocked reflection for pattern score test"}

        class MockCNNPatternsForPatternScoreTest(CNNPatterns):
            def get_all_detectable_pattern_keys(self) -> List[str]:
                return ["bullFlag", "bearTrap", "futurePattern"]

            async def detect_patterns_multi_timeframe(self, candles_by_timeframe: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
                return {
                    "cnn_predictions": {
                        "5m_bullFlag_score": 0.85,  # Above threshold 0.7
                        "1h_bullFlag_score": 0.6,   # Below threshold
                        "5m_bearTrap_score": 0.5,   # Below threshold
                        "15m_nonCnnPattern_score": 0.9, # Not in get_all_detectable_pattern_keys
                        "1h_futurePattern_score": 0.7 # At threshold
                    },
                    "patterns": {}, # Keep other parts minimal
                    "context": {}
                }

        # Need MockBiasReflector and MockConfidenceEngine from the outer scope or define them here
        # They are defined in run_test_ai_activation_engine, so we can reuse them if this test is called from there,
        # or define simplified versions here. For standalone clarity, let's define simple ones.
        class MockBiasReflector:
            def get_bias_score(self, token, strategy_id): return 0.5
        class MockConfidenceEngine:
            def get_confidence_score(self, token, strategy_id): return 0.5


        mock_reflectie_lus_pattern_test = MockReflectieLusForTestPatternScore()
        engine = AIActivationEngine(reflectie_lus_instance=mock_reflectie_lus_pattern_test)

        # Assign the specialized mock CNN detector
        engine.cnn_patterns_detector = MockCNNPatternsForPatternScoreTest()

        # Explicitly set the threshold on the instance for this test, mirroring class constant
        # This ensures the test uses the intended value even if class structure changes.
        engine.CNN_DETECTION_THRESHOLD = 0.7

        calculated_pattern_score = None
        with patch.object(engine, '_should_trigger_ai', new_callable=AsyncMock, return_value=False) as mock_should_trigger:
            await engine.activate_ai(
                trigger_type='test_pattern_score_trigger',
                token='TEST/PATTERNSCORE',
                candles_by_timeframe={}, # Mocked, not used by MockCNNPatternsForPatternScoreTest
                strategy_id='test_pattern_score_strat',
                bias_reflector_instance=MockBiasReflector(),
                confidence_engine_instance=MockConfidenceEngine(),
                mode='test' # ensure it doesn't force trigger via 'pretrain' or 'backtest_reflection'
            )

            assert mock_should_trigger.called, "_should_trigger_ai was not called, pattern_score_val might not have been calculated or passed."

            # Retrieve the 'trigger_data' argument from the call to _should_trigger_ai
            # call_args is a tuple: (args, kwargs). args is a tuple of positional arguments.
            called_args = mock_should_trigger.call_args[0]
            assert len(called_args) > 0, "_should_trigger_ai called without arguments."
            trigger_data_arg = called_args[0] # The first argument is trigger_data

            calculated_pattern_score = trigger_data_arg.get('patternScore')

        assert calculated_pattern_score is not None, "patternScore was not found in trigger_data."

        # Expected: bullFlag (0.85 >= 0.7) is 1, futurePattern (0.7 >= 0.7) is 1. Total 2.
        # Total possible patterns = 3 (bullFlag, bearTrap, futurePattern)
        # Score = 2 / 3
        expected_score = 2/3
        assert abs(calculated_pattern_score - expected_score) < 0.001, \
            f"Calculated pattern_score_val {calculated_pattern_score} does not match expected {expected_score}"

        logger.info(f"Test pattern_score_val calculation: PASSED. Score: {calculated_pattern_score:.4f} (Expected: {expected_score:.4f})")
        logger.info("--- Test pattern_score_val Calculation Complete ---")


    async def test_should_trigger_ai_logic(engine_instance: AIActivationEngine):
        logger.info("\n--- Test _should_trigger_ai Logic ---")

        # Test case 1: Sentiment data found
        with patch.object(engine_instance.grok_sentiment_fetcher, 'fetch_live_search_data', new_callable=AsyncMock, return_value=[{"item": "data"}]):
            trigger_data_sentiment = {"patternScore": 0.5, "volumeSpike": False, "learned_confidence": 0.7, "time_since_last_reflection": 0, "profit_metric": 0.0}
            # Expected score: 0 (base) + 1 (sentiment) = 1. Confidence > 0.6 no points. time_since_last no points, profit_metric no points.
            # Threshold for confidence 0.7 is 3. So, 1 < 3, should be False.
            # Let's make learned_confidence low to trigger
            trigger_data_sentiment_low_conf = {"patternScore": 0.5, "volumeSpike": False, "learned_confidence": 0.3, "time_since_last_reflection": 0, "profit_metric": 0.0}
            # Score: 1 (sentiment) + 2 (low confidence) = 3. Threshold for conf 0.3 is 2. 3 >= 2, should be True.
            result_sentiment = await engine_instance._should_trigger_ai(trigger_data_sentiment_low_conf, "TEST/TOKEN", "live")
            assert result_sentiment is True, "Should trigger with sentiment and low confidence"
            logger.info(f"Test with sentiment data (low conf): Triggered = {result_sentiment} (Expected True)")

        # Test case 2: No sentiment data
        with patch.object(engine_instance.grok_sentiment_fetcher, 'fetch_live_search_data', new_callable=AsyncMock, return_value=[]):
            trigger_data_no_sentiment_low_conf = {"patternScore": 0.5, "volumeSpike": False, "learned_confidence": 0.3, "time_since_last_reflection": 0, "profit_metric": 0.0}
            # Score: 0 (no sentiment) + 2 (low confidence) = 2. Threshold for conf 0.3 is 2. 2 >= 2, should be True.
            result_no_sentiment = await engine_instance._should_trigger_ai(trigger_data_no_sentiment_low_conf, "TEST/TOKEN", "live")
            assert result_no_sentiment is True, "Should trigger with low confidence even without sentiment" # This depends on threshold logic
            logger.info(f"Test with no sentiment data (low conf): Triggered = {result_no_sentiment} (Expected True)")

            # Test case 2b: No sentiment, high confidence (should not trigger)
            trigger_data_no_sentiment_high_conf = {"patternScore": 0.1, "volumeSpike": False, "learned_confidence": 0.8, "time_since_last_reflection": 0, "profit_metric": 0.0}
            # Score: 0 (no sentiment). Threshold for conf 0.8 is 3. 0 < 3, should be False.
            result_no_sentiment_high_conf = await engine_instance._should_trigger_ai(trigger_data_no_sentiment_high_conf, "TEST/TOKEN", "live")
            assert result_no_sentiment_high_conf is False, "Should NOT trigger with no sentiment and high confidence and low pattern score"
            logger.info(f"Test with no sentiment data (high conf, low pattern): Triggered = {result_no_sentiment_high_conf} (Expected False)")

        # Test case 3: GrokSentimentFetcher is None (simulating initialization failure)
        # Ensure GrokSentimentFetcher might be None in __init__ for this test to be fully valid.
        # It is, due to the try-except block.
        original_fetcher = engine_instance.grok_sentiment_fetcher
        engine_instance.grok_sentiment_fetcher = None
        trigger_data_no_fetcher_low_conf = {"patternScore": 0.5, "volumeSpike": False, "learned_confidence": 0.3, "time_since_last_reflection": 0, "profit_metric": 0.0}
        # Score: 0 (no sentiment fetcher) + 2 (low confidence) = 2. Threshold for conf 0.3 is 2. 2 >= 2, should be True.
        result_no_fetcher = await engine_instance._should_trigger_ai(trigger_data_no_fetcher_low_conf, "TEST/TOKEN", "live")
        assert result_no_fetcher is True, "Should trigger with low confidence even if fetcher is None"
        logger.info(f"Test with GrokSentimentFetcher as None (low conf): Triggered = {result_no_fetcher} (Expected True)")
        engine_instance.grok_sentiment_fetcher = original_fetcher # Restore

        logger.info("--- _should_trigger_ai Logic Test Complete ---")

    asyncio.run(run_test_ai_activation_engine())
