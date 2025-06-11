# core/entry_decider.py
import logging
from typing import Dict, Any, Optional
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import dotenv
import json

# Importeer AI-modules
from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.cnn_patterns import CNNPatterns
from core.params_manager import ParamsManager
from core.cooldown_tracker import CooldownTracker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EntryDecider:
    """
    Neemt AI-gestuurde entry-besluiten op basis van consensus, confidence, bias en patronen.
    """

    def __init__(self):
        self.gpt_reflector = GPTReflector()
        self.grok_reflector = GrokReflector()
        try:
            self.prompt_builder = PromptBuilder()
        except Exception as e:
            logger.error(f"Failed to initialize PromptBuilder: {e}")
            self.prompt_builder = None
        self.bias_reflector = BiasReflector()
        try:
            self.confidence_engine = ConfidenceEngine()
        except Exception as e:
            logger.error(f"Failed to initialize ConfidenceEngine: {e}")
            self.confidence_engine = None
        self.cnn_patterns_detector = CNNPatterns()
        self.params_manager = ParamsManager()
        self.cooldown_tracker = CooldownTracker() # Added CooldownTracker
        logger.info("EntryDecider geïnitialiseerd.")

    async def get_consensus(self, prompt: str, token: str, strategy_id: str, current_learned_bias: float, current_learned_confidence: float) -> Dict[str, Any]:
        """
        Vraagt AI-modellen om input en combineert hun oordelen voor consensus.
        """
        logger.debug(f"Vragen om AI-consensus voor {token} ({strategy_id})...")

        gpt_response = await self.gpt_reflector.ask_ai(prompt, context={"token": token, "strategy_id": strategy_id, "learned_bias": current_learned_bias, "learned_confidence": current_learned_confidence})
        grok_response = await self.grok_reflector.ask_grok(prompt, context={"token": token, "strategy_id": strategy_id, "learned_bias": current_learned_bias, "learned_confidence": current_learned_confidence})

        gpt_confidence = gpt_response.get('confidence', 0.0) or 0.0
        grok_confidence = grok_response.get('confidence', 0.0) or 0.0
        gpt_intentie = str(gpt_response.get('intentie', '')).upper()
        grok_intentie = str(grok_response.get('intentie', '')).upper()

        consensus_intentie = "HOLD"
        if gpt_intentie == grok_intentie and gpt_intentie in ["LONG", "SHORT"]:
            consensus_intentie = gpt_intentie
        elif gpt_intentie in ["LONG", "SHORT"] and (not grok_intentie or grok_intentie == "HOLD"):
            consensus_intentie = gpt_intentie
        elif grok_intentie in ["LONG", "SHORT"] and (not gpt_intentie or gpt_intentie == "HOLD"):
            consensus_intentie = grok_intentie
        elif gpt_intentie in ["LONG", "SHORT"] and grok_intentie in ["LONG", "SHORT"] and gpt_intentie != grok_intentie:
            # If intentions conflict, default to HOLD. More sophisticated handling could be added.
            consensus_intentie = "HOLD" # Or consider highest confidence, or other tie-breaking
            logger.info(f"Conflicting AI intentions for {token}: GPT={gpt_intentie}, Grok={grok_intentie}. Defaulting to HOLD.")

        # Calculate combined confidence, ensuring only valid numbers are used
        num_valid_confidences = sum(1 for conf in [gpt_confidence, grok_confidence] if isinstance(conf, (float, int)) and conf >= 0)
        if num_valid_confidences > 0:
            combined_confidence = (gpt_confidence + grok_confidence) / num_valid_confidences
        else:
            combined_confidence = 0.0 # Default if no valid confidences

        # Handle reported bias from AI responses, fallback to learned_bias if invalid
        gpt_reported_bias = gpt_response.get('bias', current_learned_bias)
        if not isinstance(gpt_reported_bias, (float, int)): gpt_reported_bias = current_learned_bias

        grok_reported_bias = grok_response.get('bias', current_learned_bias)
        if not isinstance(grok_reported_bias, (float, int)): grok_reported_bias = current_learned_bias

        combined_bias_reported = (gpt_reported_bias + grok_reported_bias) / 2.0


        logger.info(f"AI Consensus voor {token}: Intentie={consensus_intentie}, Confidence={combined_confidence:.2f}, Reported Bias={combined_bias_reported:.2f}")

        return {
            "consensus_intentie": consensus_intentie,
            "combined_confidence": combined_confidence,
            "combined_bias_reported": combined_bias_reported, # This is the bias *reported by AI*, not necessarily the strategy's learned_bias
            "gpt_raw": gpt_response,
            "grok_raw": grok_response
        }


    async def should_enter(
        self,
        dataframe: pd.DataFrame, # Base timeframe DataFrame
        symbol: str,
        current_strategy_id: str,
        trade_context: Optional[Dict[str, Any]] = None, # Includes candles_by_timeframe, current_price etc.
        # De volgende zijn leerbare parameters, doorgegeven vanuit de strategie
        learned_bias: float = 0.5, # Current learned bias for the symbol/strategy
        learned_confidence: float = 0.5, # Current learned confidence for the symbol/strategy
        entry_conviction_threshold: float = 0.7 # Minimum AI confidence to consider entry
    ) -> Dict[str, Any]:
        """
        Bepaalt of een entry moet worden geplaatst op basis van AI-consensus en drempelwaarden.
        """
        logger.debug(f"[EntryDecider] Evalueren entry voor {symbol} met strategie {current_strategy_id}...")

        if dataframe.empty:
            logger.warning(f"[EntryDecider] Geen dataframe beschikbaar voor {symbol}. Kan geen entry besluit nemen.")
            return {"enter": False, "reason": "no_dataframe", "confidence": 0, "learned_bias": learned_bias, "ai_intent": "HOLD", "pattern_details": {}}

        # --- AI-specifieke Cooldown Check ---
        if self.cooldown_tracker.is_cooldown_active(symbol, current_strategy_id):
            cooldown_info = self.cooldown_tracker._cooldown_state.get(symbol, {}).get(current_strategy_id, {})
            cooldown_reason = cooldown_info.get('reason', 'unknown')
            cooldown_end_time_str = cooldown_info.get('end_time', 'N/A')
            logger.info(f"Entry geweigerd voor {symbol} door AI-specifieke cooldown (reden: {cooldown_reason}, eindigt: {cooldown_end_time_str}).")
            return {"enter": False, "reason": f"ai_cooldown_active_{cooldown_reason}", "confidence": learned_confidence, "learned_bias": learned_bias, "ai_intent": "HOLD", "pattern_details": {}}


        # --- Time-of-Day Effectiveness Check ---
        current_hour = datetime.now().hour # Use current time for this check
        # Fetch timeOfDayEffectiveness from params_manager (expected to be a dict like {"0": 0.2, "1": -0.1, ...})
        time_effectiveness_data = self.params_manager.get_param("timeOfDayEffectiveness", strategy_id=None) # Global parameter

        hour_effectiveness = 0.0 # Default to neutral if not found or data is malformed
        if isinstance(time_effectiveness_data, dict):
            hour_effectiveness = time_effectiveness_data.get(str(current_hour), 0.0)
        else:
            logger.warning(f"timeOfDayEffectiveness data is not a dict or not found: {time_effectiveness_data}. Using neutral (0.0).")

        # Adjust AI consensus confidence based on time-of-day effectiveness
        # Positive effectiveness increases confidence, negative decreases it.
        # The multiplier's sensitivity can be tuned (e.g., * 0.5 means a score of 1.0 or -1.0 from effectiveness changes confidence by +/-50%)
        time_adjusted_confidence_multiplier = 1.0 + (hour_effectiveness * 0.5)
        time_adjusted_confidence_multiplier = max(0.1, min(time_adjusted_confidence_multiplier, 2.0)) # Clamp multiplier (e.g., 0.1x to 2.0x)

        logger.debug(f"Time-of-day effectiveness for hour {current_hour}: {hour_effectiveness:.2f}. Base Confidence Multiplier: {time_adjusted_confidence_multiplier:.2f}.")


        # Genereer prompt voor AI
        # Ensure candles_by_timeframe is properly constructed, including the base dataframe
        candles_by_timeframe = trade_context.get('candles_by_timeframe', {}) if trade_context else {}
        base_tf_name = dataframe.attrs.get('timeframe', 'unknown_tf') # Get timeframe name from df attributes
        if base_tf_name not in candles_by_timeframe and not dataframe.empty:
             candles_by_timeframe[base_tf_name] = dataframe.copy()


        prompt = None
        if self.prompt_builder:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe,
                symbol=symbol,
                prompt_type='marketAnalysis', # Or 'entrySignal'
                current_bias=learned_bias,
                current_confidence=learned_confidence
            )
        else:
            logger.error("[EntryDecider] PromptBuilder not available. Cannot generate prompt for entry decision.")
            return {"enter": False, "reason": "prompt_builder_unavailable", "confidence": learned_confidence, "learned_bias": learned_bias, "ai_intent": "HOLD", "pattern_details": {}}

        if not prompt: # Should be redundant if PromptBuilder error is caught, but as a safeguard
            logger.warning(f"[EntryDecider] Geen prompt gegenereerd voor {symbol}. Entry geweigerd.")
            return {"enter": False, "reason": "no_prompt_generated", "confidence": learned_confidence, "learned_bias": learned_bias, "ai_intent": "HOLD", "pattern_details": {}}

        # Vraag AI-consensus
        consensus_result = await self.get_consensus(prompt, symbol, current_strategy_id, learned_bias, learned_confidence)
        consensus_intentie = consensus_result['consensus_intentie']
        ai_combined_confidence = consensus_result['combined_confidence']

        # Pas AI-consensus confidence aan met time-of-day multiplier
        final_ai_confidence = ai_combined_confidence * time_adjusted_confidence_multiplier
        final_ai_confidence = min(final_ai_confidence, 1.0) # Ensure it doesn't exceed 1.0 (max confidence)
        final_ai_confidence = max(final_ai_confidence, 0.0) # Ensure it doesn't go below 0.0 (min confidence)

        logger.info(f"Symbol: {symbol}, Initial AI Conf: {ai_combined_confidence:.2f}, Time Multiplier: {time_adjusted_confidence_multiplier:.2f}, Final AI Conf: {final_ai_confidence:.2f}")


        # CNN patroon check
        pattern_data = await self.cnn_patterns_detector.detect_patterns_multi_timeframe(candles_by_timeframe, symbol)
        has_strong_cnn_pattern = False
        # cnnPatternWeight = self.params_manager.get_param("cnnPatternWeight", strategy_id=current_strategy_id) # Fetched but not used yet

        if pattern_data and pattern_data.get('patterns'):
            bullish_patterns = ['bullishEngulfing', 'CDLENGULFING', 'morningStar', 'CDLMORNINGSTAR',
                                'threeWhiteSoldiers', 'CDL3WHITESOLDIERS', 'bullFlag', 'bullishFractal',
                                'bullishRSIDivergence', 'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLPIERCING', 'ascendingTriangle', 'pennant']
            # Check if any of the detected patterns (keys in pattern_data['patterns']) are in our bullish_patterns list
            if any(p.upper() in (key.upper() for key in pattern_data['patterns'].keys()) for p in bullish_patterns):
                has_strong_cnn_pattern = True
                logger.info(f"Sterk bullish CNN patroon gedetecteerd voor {symbol}: {pattern_data['patterns']}")

        # AI-besluitvormingslogica
        # Entry conditions: AI intent is LONG, final (time-adjusted) confidence meets threshold,
        # learned bias is sufficiently bullish, and a strong CNN pattern is present.
        if consensus_intentie == "LONG" and \
           final_ai_confidence >= entry_conviction_threshold and \
           learned_bias >= 0.55 and \
           has_strong_cnn_pattern:

            logger.info(f"[EntryDecider] ✅ Entry GOEDKEURING voor {symbol}. Consensus: {consensus_intentie}, Final AI Conf: {final_ai_confidence:.2f} (Threshold: {entry_conviction_threshold:.2f}), Geleerde Bias: {learned_bias:.2f} (Threshold: >=0.55), Patroon: {has_strong_cnn_pattern}.")
            return {
                "enter": True,
                "reason": "AI_CONSENSUS_LONG_CONDITIONS_MET",
                "confidence": final_ai_confidence,
                "learned_bias": learned_bias,
                "ai_intent": consensus_intentie,
                "ai_details": consensus_result, # Full AI consensus details
                "pattern_details": pattern_data.get('patterns', {})
            }
        else:
            # Construct detailed reason for rejection
            reason_parts = []
            if consensus_intentie != "LONG": reason_parts.append(f"ai_intent_not_long ({consensus_intentie})")
            if final_ai_confidence < entry_conviction_threshold: reason_parts.append(f"final_ai_conf_low ({final_ai_confidence:.2f} < {entry_conviction_threshold:.2f})")
            if learned_bias < 0.55: reason_parts.append(f"learned_bias_low ({learned_bias:.2f} < 0.55)")
            if not has_strong_cnn_pattern: reason_parts.append("no_strong_bullish_cnn_pattern")

            full_reason_str = "_".join(reason_parts) if reason_parts else "entry_conditions_not_met"
            if not reason_parts and consensus_intentie == "LONG": # If intent was long but other conditions failed
                 full_reason_str = f"intent_long_other_conditions_failed_conf{final_ai_confidence:.2f}_bias{learned_bias:.2f}_pattern{has_strong_cnn_pattern}"


            logger.info(f"[EntryDecider] ❌ Entry GEWEIGERD voor {symbol}. Reden: {full_reason_str}. AI Intentie: {consensus_intentie}, Final AI Conf: {final_ai_confidence:.2f}, Geleerde Bias: {learned_bias:.2f}, Patroon: {has_strong_cnn_pattern}.")
            return {
                "enter": False,
                "reason": full_reason_str,
                "confidence": final_ai_confidence, # Report final (potentially adjusted) confidence
                "learned_bias": learned_bias,
                "ai_intent": consensus_intentie,
                "ai_details": consensus_result,
                "pattern_details": pattern_data.get('patterns', {})
            }

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Mock Freqtrade DataFrame (same as in cnn_patterns test)
    def create_mock_dataframe_for_entry_decider(timeframe: str = '5m', num_candles: int = 100) -> pd.DataFrame:
        data = []
        now = datetime.utcnow() # Use utcnow for consistency
        interval_seconds_map = { # Expanded map
            '1m': 60, '5m': 300, '15m': 900, '1h': 3600,
            '4h': 14400, '12h': 43200, '1d': 86400
        }
        interval_seconds = interval_seconds_map.get(timeframe, 300)


        for i in range(num_candles):
            date = now - timedelta(seconds=(num_candles - 1 - i) * interval_seconds)
            open_ = 100 + i * 0.1 + np.random.rand() * 2
            close_ = open_ + (np.random.rand() - 0.5) * 5 # Allow more variation
            high_ = max(open_, close_) + np.random.rand() * 2
            low_ = min(open_, close_) - np.random.rand() * 2
            volume = 1000 + np.random.rand() * 500

            # Mock indicators (can be more sophisticated if needed)
            rsi = 50 + (np.random.rand() - 0.5) * 30
            macd_val = (np.random.rand() - 0.5) * 0.1
            macdsignal_val = macd_val * (0.8 + np.random.rand()*0.2)
            macdhist_val = macd_val - macdsignal_val

            # Simplified Bollinger Bands for mock data
            sma_period = 20
            std_dev_multiplier = 2
            # current_candle_data_for_sma = [] # Not used
            if i >= sma_period -1 and len(data) >= sma_period -1: # check i against sma_period-1
                recent_closes_for_bb = [r[4] for r in data[-(sma_period-1):]] + [close_] # use index 4 for close
                if len(recent_closes_for_bb) == sma_period: # Ensure we have exactly sma_period points
                    sma = np.mean(recent_closes_for_bb)
                    std_dev = np.std(recent_closes_for_bb)
                    bb_middle = sma
                    bb_upper = sma + std_dev_multiplier * std_dev
                    bb_lower = sma - std_dev_multiplier * std_dev
                else: # Fallback if not enough data yet for full SMA
                    bb_middle = open_
                    bb_upper = high_
                    bb_lower = low_
            else: # Fallback for initial candles
                bb_middle = open_
                bb_upper = high_
                bb_lower = low_

            data.append([date, open_, high_, low_, close_, volume, rsi, macd_val, macdsignal_val, macdhist_val, bb_upper, bb_middle, bb_lower])

        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_upperband', 'bb_middleband', 'bb_lowerband'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.attrs['timeframe'] = timeframe
        df.attrs['pair'] = 'ETH/USDT' # Add pair attribute for context
        return df

    async def run_test_entry_decider():
        decider = EntryDecider()
        test_symbol = "ETH/USDT"
        test_strategy_id = "DUOAI_Strategy"

        # Create mock dataframes
        mock_df_5m = create_mock_dataframe_for_entry_decider(timeframe='5m', num_candles=60)
        # Make last candle bullish for pattern detection test
        last_idx = mock_df_5m.index[-1]
        mock_df_5m.loc[last_idx, 'open'] = mock_df_5m.loc[last_idx, 'low'] + 0.1
        mock_df_5m.loc[last_idx, 'close'] = mock_df_5m.loc[last_idx, 'high'] - 0.1
        mock_df_5m.loc[last_idx, 'volume'] = mock_df_5m['volume'].mean() * 3

        mock_candles_by_timeframe = {
            '5m': mock_df_5m,
            '1h': create_mock_dataframe_for_entry_decider('1h', 60)
        }
        mock_trade_context = {"stake_amount": 100, "candles_by_timeframe": mock_candles_by_timeframe, "current_price": mock_df_5m['close'].iloc[-1]}


        # Store original methods to restore after test
        original_prompt_builder_generate = decider.prompt_builder.generate_prompt_with_data if decider.prompt_builder else None
        original_gpt_ask = decider.gpt_reflector.ask_ai
        original_grok_ask = decider.grok_reflector.ask_grok
        original_cnn_detect = decider.cnn_patterns_detector.detect_patterns_multi_timeframe
        original_bias_get = decider.bias_reflector.get_bias_score
        original_conf_get = decider.confidence_engine.get_confidence_score if decider.confidence_engine else None
        original_params_get = decider.params_manager.get_param
        original_cooldown_active = decider.cooldown_tracker.is_cooldown_active


        # --- Test Scenario 1: Positive Entry (Good Time of Day) ---
        print("\n--- Test EntryDecider (Positief Scenario, Goed Uur) ---")
        # Mock dependencies for a positive entry
        if decider.prompt_builder:
            async def mock_generate_prompt_positive(*args, **kwargs): return "Mock prompt for positive entry."
            decider.prompt_builder.generate_prompt_with_data = mock_generate_prompt_positive
        async def mock_ask_ai_positive(*args, **kwargs): return {"reflectie": "AI: Markt veelbelovend.", "confidence": 0.85, "intentie": "LONG", "emotie": "optimistisch", "bias": 0.7}
        decider.gpt_reflector.ask_ai = mock_ask_ai_positive
        decider.grok_reflector.ask_ai = mock_ask_ai_positive # Simplified: Grok agrees
        async def mock_cnn_bullish(*args, **kwargs): return {"patterns": {"bullishEngulfing": True, "CDLENGULFING": True}, "context": {"trend": "uptrend"}}
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_bullish

        # Mock params_manager for positive time-of-day effectiveness
        def mock_params_get_positive_time(key, strategy_id=None):
            if key == "timeOfDayEffectiveness": return {str(datetime.now().hour): 0.5} # Positive effectiveness
            if key == "entryConvictionThreshold": return 0.7
            if key == "cnnPatternWeight": return 1.0 # Example value
            return 0.0 # Default
        decider.params_manager.get_param = mock_params_get_positive_time
        decider.cooldown_tracker.is_cooldown_active = lambda t, s: False # Ensure no cooldown

        entry_decision_positive = await decider.should_enter(
            dataframe=mock_df_5m, symbol=test_symbol, current_strategy_id=test_strategy_id,
            trade_context=mock_trade_context,
            learned_bias=0.7, learned_confidence=0.8, entry_conviction_threshold=0.7
        )
        print("Resultaat (Positief Scenario):", json.dumps(entry_decision_positive, indent=2, default=str))
        assert entry_decision_positive['enter'] is True
        assert "AI_CONSENSUS_LONG_CONDITIONS_MET" in entry_decision_positive['reason']
        assert entry_decision_positive['confidence'] > 0.8 # Expected to be boosted by time effectiveness

        # --- Test Scenario 2: Negative Time-of-Day Effectiveness Blocks Entry ---
        print("\n--- Test EntryDecider (Negatief Uur Blokkeert Entry) ---")
        # Keep AI positive, but time of day is bad
        decider.gpt_reflector.ask_ai = mock_ask_ai_positive # Re-assign as it might have been changed by other tests if run in parallel
        decider.grok_reflector.ask_ai = mock_ask_ai_positive
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_bullish

        def mock_params_get_negative_time(key, strategy_id=None):
            if key == "timeOfDayEffectiveness": return {str(datetime.now().hour): -0.8} # Strong negative effectiveness
            if key == "entryConvictionThreshold": return 0.7
            if key == "cnnPatternWeight": return 1.0
            return 0.0
        decider.params_manager.get_param = mock_params_get_negative_time

        entry_decision_neg_time = await decider.should_enter(
            dataframe=mock_df_5m, symbol=test_symbol, current_strategy_id=test_strategy_id,
            trade_context=mock_trade_context,
            learned_bias=0.7, learned_confidence=0.8, entry_conviction_threshold=0.7
        )
        print("Resultaat (Negatief Uur):", json.dumps(entry_decision_neg_time, indent=2, default=str))
        assert entry_decision_neg_time['enter'] is False
        assert "final_ai_conf_low" in entry_decision_neg_time['reason'] # Confidence should be reduced below threshold

        # --- Test Scenario 3: Cooldown Active Blocks Entry ---
        print("\n--- Test EntryDecider (Cooldown Actief Blokkeert Entry) ---")
        decider.cooldown_tracker.is_cooldown_active = lambda t, s: True # Mock cooldown as active
        # Ensure params_manager is reset to a state that would otherwise allow entry
        decider.params_manager.get_param = mock_params_get_positive_time

        entry_decision_cooldown = await decider.should_enter(
            dataframe=mock_df_5m, symbol=test_symbol, current_strategy_id=test_strategy_id,
            trade_context=mock_trade_context,
            learned_bias=0.7, learned_confidence=0.8, entry_conviction_threshold=0.7
        )
        print("Resultaat (Cooldown Actief):", json.dumps(entry_decision_cooldown, indent=2, default=str))
        assert entry_decision_cooldown['enter'] is False
        assert "ai_cooldown_active" in entry_decision_cooldown['reason']


        # Restore original methods
        if decider.prompt_builder:
            decider.prompt_builder.generate_prompt_with_data = original_prompt_builder_generate
        decider.gpt_reflector.ask_ai = original_gpt_ask
        decider.grok_reflector.ask_grok = original_grok_ask
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = original_cnn_detect
        decider.bias_reflector.get_bias_score = original_bias_get
        if decider.confidence_engine:
            decider.confidence_engine.get_confidence_score = original_conf_get
        decider.params_manager.get_param = original_params_get
        decider.cooldown_tracker.is_cooldown_active = original_cooldown_active


    asyncio.run(run_test_entry_decider())
