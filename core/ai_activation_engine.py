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
# ReflectieLus, BiasReflector, ConfidenceEngine, StrategyManager
# will be passed via dependency injection to activate_ai to avoid circular imports.

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

    def __init__(self):
        try:
            self.prompt_builder = PromptBuilder()
        except Exception as e:
            logger.error(f"Failed to initialize PromptBuilder in AIActivationEngine: {e}")
            self.prompt_builder = None # Fallback

        self.gpt_reflector = GPTReflector()
        self.grok_reflector = GrokReflector()
        self.cnn_patterns_detector = CNNPatterns()

        self.last_reflection_timestamps: Dict[str, float] = {} # Per token
        logger.info("AIActivationEngine geïnitialiseerd.")

    def _get_time_since_last_reflection(self, token: str) -> float:
        return (datetime.now().timestamp() * 1000) - self.last_reflection_timestamps.get(token, 0)

    def _update_last_reflection_timestamp(self, token: str):
        self.last_reflection_timestamps[token] = datetime.now().timestamp() * 1000

    def _should_trigger_ai(self, trigger_data: Dict[str, Any], mode: str = 'live') -> bool:
        """
        Bepaalt of de AI-reflectie moet worden getriggerd op basis van verschillende factoren.
        """
        score = 0
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

        pattern_score_val = 0.0
        if pattern_data and pattern_data.get('patterns'):
            detected_pattern_count = sum(1 for v in pattern_data['patterns'].values() if v)
            # This method needs to be implemented in CNNPatterns to return a list of all pattern keys it can detect.
            # total_possible_patterns = len(self.cnn_patterns_detector.get_all_detectable_pattern_keys())
            # For now, using a placeholder or assuming all keys in 'patterns' are relevant if present.
            all_pattern_keys_in_result = len(pattern_data['patterns']) # Number of patterns checked in this run
            total_possible_patterns = all_pattern_keys_in_result if all_pattern_keys_in_result > 0 else 1 # Avoid div by zero
            # A more robust way would be to have a fixed list of all pattern types the system knows.
            # total_possible_patterns = len(['bullFlag', 'bearFlag', ...]) # Example
            if total_possible_patterns > 0:
                pattern_score_val = detected_pattern_count / total_possible_patterns

        trigger_data = {
            "patternScore": pattern_score_val,
            "volumeSpike": pattern_data.get('context', {}).get('volume_spike', False),
            "learned_confidence": learned_confidence,
            "time_since_last_reflection": self._get_time_since_last_reflection(token),
            "profit_metric": trade_context.get('profit_pct', 0.0) if trade_context else 0.0
        }

        if not self._should_trigger_ai(trigger_data, mode):
            logger.info(f"[AI-ACTIVATION] AI niet getriggerd voor {token} ({trigger_type}).")
            return None

        prompt = await self.prompt_builder.generate_prompt_with_data(
            candles_by_timeframe=candles_by_timeframe,
            symbol=token,
            prompt_type='comprehensive_analysis',
            current_bias=current_bias,
            current_confidence=learned_confidence,
            trade_context=trade_context,
            trigger_type=trigger_type,
            pattern_data=pattern_data
        )
        if not prompt:
            logger.warning(f"[AI-ACTIVATION] Geen prompt gegenereerd voor {token}.")
            return None

        gpt_result = await self.gpt_reflector.ask_ai(prompt, context={"token": token, "strategy_id": strategy_id, "trigger_type": trigger_type, **(trade_context or {})})
        grok_result = await self.grok_reflector.ask_grok(prompt, context={"token": token, "strategy_id": strategy_id, "trigger_type": trigger_type, **(trade_context or {})})

        gpt_confidence = gpt_result.get('confidence', 0.0) or 0.0
        grok_confidence = grok_result.get('confidence', 0.0) or 0.0

        num_valid_confidences = sum(1 for c in [gpt_confidence, grok_confidence] if isinstance(c, (float,int)) and c > 0)
        combined_ai_confidence = ((gpt_confidence + grok_confidence) / num_valid_confidences) if num_valid_confidences > 0 else 0.0

        gpt_reported_bias = gpt_result.get('bias', current_bias)
        grok_reported_bias = grok_result.get('bias', current_bias)
        combined_ai_reported_bias = ((gpt_reported_bias or current_bias) + (grok_reported_bias or current_bias)) / 2.0

        reflectie_log_entry = {
            "timestamp": datetime.now().isoformat(),
            "trigger_type": trigger_type,
            "token": token,
            "strategyId": strategy_id,
            "pattern_data": pattern_data,
            "trigger_decision_data": trigger_data,
            "prompt_generated": prompt,
            "gpt_response": gpt_result,
            "grok_response": grok_result,
            "trade_context": trade_context,
            "learned_bias_at_trigger": current_bias,
            "learned_confidence_at_trigger": learned_confidence,
            "combined_ai_confidence": combined_ai_confidence,
            "combined_ai_reported_bias": combined_ai_reported_bias,
            "mode": mode
        }

        if mode not in ['pretrain', 'backtest_simulation']:
            await self._log_reflection_entry(reflectie_log_entry)
            self._update_last_reflection_timestamp(token)
            logger.info(f"[AI-ACTIVATION] ✅ Reflectie gegenereerd en gelogd voor {token} ({trigger_type}).")

        return reflectie_log_entry

if __name__ == "__main__":
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
        engine = AIActivationEngine()
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

        if engine.prompt_builder is None:
            class MockPromptBuilder:
                async def generate_prompt_with_data(self, **kwargs):
                    return f"Mock prompt for {kwargs.get('symbol')} - {kwargs.get('prompt_type')}"
            engine.prompt_builder = MockPromptBuilder()

        print("\n--- Test AIActivationEngine ---")
        print("\nActiveren AI voor een 'entry_signal' trigger...")
        entry_reflection = await engine.activate_ai(
            trigger_type='entry_signal',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context={"signal_strength": 0.85},
            mode='dry_run',
            bias_reflector_instance=mock_bias_reflector,
            confidence_engine_instance=mock_confidence_engine
        )
        if entry_reflection:
            print("\nResultaat Entry Signaal Reflectie:", json.dumps(entry_reflection, indent=2, default=str))
        else:
            print("Entry Signaal Reflectie niet getriggerd / geen resultaat.")

        print("\nActiveren AI voor een 'trade_closed' trigger...")
        exit_reflection = await engine.activate_ai(
            trigger_type='trade_closed',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context={"entry_price": 2500, "exit_price": 2450, "profit_pct": -0.02, "trade_id": "t123"},
            mode='live',
            bias_reflector_instance=mock_bias_reflector,
            confidence_engine_instance=mock_confidence_engine
        )
        if exit_reflection:
            print("\nResultaat Trade Closed Reflectie:", json.dumps(exit_reflection, indent=2, default=str))
        else:
            print("Trade Closed Reflectie niet getriggerd / geen resultaat.")

        print(f"Check {REFLECTIE_LOG_FILE} voor gelogde reflecties.")

    asyncio.run(run_test_ai_activation_engine())
