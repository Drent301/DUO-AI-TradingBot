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
from core.params_manager import ParamsManager # Restored
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
        self.params_manager = ParamsManager() # Restored
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
        # cnnPatternWeight is now fetched from ParamsManager
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
        # Fetch timeOfDayEffectiveness from params_manager
        time_effectiveness_data = self.params_manager.get_param("timeOfDayEffectiveness", strategy_id=None)
        if time_effectiveness_data is None:
            time_effectiveness_data = {} # Default to empty dict if not found
            logger.warning("timeOfDayEffectiveness not found in ParamsManager, using default empty {}.")


        hour_effectiveness = 0.0 # Default to neutral if not found or data is malformed
        if isinstance(time_effectiveness_data, dict):
            hour_effectiveness = time_effectiveness_data.get(str(current_hour), 0.0)
        else:
            logger.warning(f"timeOfDayEffectiveness data is not a dict or not found: {time_effectiveness_data}. Using neutral (0.0).")

        # Adjust AI consensus confidence based on time-of-day effectiveness
        # Positive effectiveness increases confidence, negative decreases it.
        entry_time_effectiveness_impact_factor = self.params_manager.get_param(
            "entryTimeEffectivenessImpactFactor",
            strategy_id=current_strategy_id,
            default=0.5
        )
        logger.info(f"Symbol: {symbol}, Using entryTimeEffectivenessImpactFactor: {entry_time_effectiveness_impact_factor} (Default: 0.5)")

        time_adjusted_confidence_multiplier = 1.0 + (hour_effectiveness * entry_time_effectiveness_impact_factor)
        time_adjusted_confidence_multiplier = max(0.1, min(time_adjusted_confidence_multiplier, 2.0)) # Clamp multiplier (e.g., 0.1x to 2.0x)

        logger.debug(f"Time-of-day effectiveness for hour {current_hour}: {hour_effectiveness:.2f}. Impact Factor: {entry_time_effectiveness_impact_factor:.2f}. Base Confidence Multiplier: {time_adjusted_confidence_multiplier:.2f}.")


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

        # Fetch cnnPatternWeight from ParamsManager
        cnn_pattern_weight = self.params_manager.get_param("cnnPatternWeight", strategy_id=current_strategy_id, default=1.0)
        if not isinstance(cnn_pattern_weight, (float, int)):
            logger.warning(f"Invalid cnnPatternWeight type ({type(cnn_pattern_weight)}) for strategy {current_strategy_id}, using default 1.0.")
            cnn_pattern_weight = 1.0
        logger.info(f"Symbol: {symbol}, Using cnnPatternWeight: {cnn_pattern_weight} (Default: 1.0)")

        weighted_pattern_score = 0.0
        detected_patterns_summary = [] # For logging
        cnn_predictions_data = pattern_data.get('cnn_predictions', {})
        rule_patterns_data = pattern_data.get('patterns', {})

        # 1. Process CNN prediction score (specifically 15m_bullFlag_score)
        # CNN predictions are assumed to be on '15m' as per cnn_patterns.py
        cnn_bull_flag_score_key = "15m_bullFlag_score"
        cnn_score = cnn_predictions_data.get(cnn_bull_flag_score_key)

        if cnn_score is not None and isinstance(cnn_score, (float, int)) and cnn_score > 0:
            contribution = cnn_score * cnn_pattern_weight
            weighted_pattern_score += contribution
            logger.info(f"Symbol: {symbol}, CNN Score contribution from '{cnn_bull_flag_score_key}': {cnn_score:.2f} * {cnn_pattern_weight:.2f} (weight) = {contribution:.2f}. Current weighted_score: {weighted_pattern_score:.2f}")
            detected_patterns_summary.append(f"CNN_{cnn_bull_flag_score_key}({cnn_score:.2f})")
        elif cnn_score is not None:
            logger.info(f"Symbol: {symbol}, CNN Score for '{cnn_bull_flag_score_key}' is {cnn_score}, not contributing to weighted_pattern_score.")
        else:
            logger.info(f"Symbol: {symbol}, No CNN Score found for '{cnn_bull_flag_score_key}'.")


        # 2. Add score for rule-based bullish patterns (break after first found)
        entry_rule_pattern_score = self.params_manager.get_param(
            "entryRulePatternScore",
            strategy_id=current_strategy_id,
            default=0.7 # This default is already correctly handled by previous change
        )
        # This log was already added when fetching entryRulePatternScore, so it's fine.
        # logger.info(f"Symbol: {symbol}, Using entryRulePatternScore: {entry_rule_pattern_score} (Default: 0.7)")

        rule_based_bullish_patterns = [
            'bullishEngulfing', 'CDLENGULFING', 'morningStar', 'CDLMORNINGSTAR',
            'threeWhiteSoldiers', 'CDL3WHITESOLDIERS', 'bullFlag', 'bullishRSIDivergence',
            'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLPIERCING', 'ascendingTriangle', 'pennant'
        ]

        if isinstance(rule_patterns_data, dict):
            for p_name in rule_based_bullish_patterns:
                if rule_patterns_data.get(p_name): # Checks for existence and truthiness
                    contribution = entry_rule_pattern_score * cnn_pattern_weight
                    weighted_pattern_score += contribution
                    logger.info(f"Symbol: {symbol}, Detected rule-based pattern: '{p_name}'. Added contribution: {entry_rule_pattern_score:.2f} * {cnn_pattern_weight:.2f} = {contribution:.2f}. Current weighted_score: {weighted_pattern_score:.2f}")
                    detected_patterns_summary.append(f"rule({p_name})")
                    break  # Only the first detected rule-based pattern contributes
        else:
            logger.warning(f"Symbol: {symbol}, 'patterns' key missing or not a dict in pattern_data. Skipping rule-based pattern check.")

        if not any(s.startswith("rule(") for s in detected_patterns_summary): # Log if no rule patterns were found and contributed
             logger.info(f"Symbol: {symbol}, No contributing rule-based bullish patterns detected from the predefined list.")


        logger.info(f"Symbol: {symbol}, Final Calculated Weighted Pattern Score: {weighted_pattern_score:.2f} from patterns: [{'; '.join(detected_patterns_summary)}]")

        # Fetch learned bias threshold from ParamsManager
        entry_learned_bias_threshold = self.params_manager.get_param(
            "entryLearnedBiasThreshold",
            strategy_id=current_strategy_id,
            default=0.55 # This default is already correctly handled by previous change
        )
        # This log was already added when fetching entryLearnedBiasThreshold, so it's fine.
        # logger.info(f"Symbol: {symbol}, Using entryLearnedBiasThreshold: {entry_learned_bias_threshold} (Default: 0.55)")

        strong_pattern_threshold_param = self.params_manager.get_param(
            "strongPatternThreshold",
            strategy_id=current_strategy_id,
            default=0.5 # Default if not found in params
        )
        logger.info(f"Symbol: {symbol}, Using strongPatternThreshold: {strong_pattern_threshold_param} (Default: 0.5)")

        # strong_pattern_threshold = entry_conviction_threshold # This line will be replaced/removed
        is_strong_pattern = weighted_pattern_score >= strong_pattern_threshold_param
        logger.info(f"Symbol: {symbol}, Weighted Pattern Score: {weighted_pattern_score:.2f}, Strong Pattern Threshold (from Params): {strong_pattern_threshold_param:.2f}, IsStrongPattern: {is_strong_pattern}")


        # AI-besluitvormingslogica
        # Entry conditions: AI intent is LONG, final (time-adjusted) confidence meets threshold,
        # learned bias is sufficiently bullish, and is_strong_pattern is True.
        if consensus_intentie == "LONG" and \
           final_ai_confidence >= entry_conviction_threshold and \
           learned_bias >= entry_learned_bias_threshold and \
           is_strong_pattern:

            logger.info(f"[EntryDecider] ✅ Entry GOEDKEURING voor {symbol}. Consensus: {consensus_intentie}, Final AI Conf: {final_ai_confidence:.2f} (Threshold: {entry_conviction_threshold:.2f}), Geleerde Bias: {learned_bias:.2f} (Threshold: >={entry_learned_bias_threshold:.2f}), Is Strong Pattern: {is_strong_pattern} (Weighted Score: {weighted_pattern_score:.2f} >= Threshold Param: {strong_pattern_threshold_param:.2f}).")
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
            if final_ai_confidence < entry_conviction_threshold: reason_parts.append(f"final_ai_conf_low ({final_ai_confidence:.2f}_vs_{entry_conviction_threshold:.2f})")
            if learned_bias < entry_learned_bias_threshold: reason_parts.append(f"learned_bias_low ({learned_bias:.2f}_vs_{entry_learned_bias_threshold:.2f})")
            if not is_strong_pattern: reason_parts.append(f"pattern_score_low ({weighted_pattern_score:.2f}_vs_{strong_pattern_threshold_param:.2f})") # Use strong_pattern_threshold_param

            full_reason_str = "_".join(reason_parts) if reason_parts else "entry_conditions_not_met"
            if not reason_parts and consensus_intentie == "LONG": # If intent was long but other conditions failed
                 full_reason_str = f"intent_long_other_conditions_failed_conf{final_ai_confidence:.2f}_bias{learned_bias:.2f}_pattern_score{weighted_pattern_score:.2f}_vs_bias_thresh{entry_learned_bias_threshold:.2f}_pattern_thresh{strong_pattern_threshold_param:.2f}" # Use strong_pattern_threshold_param


            logger.info(f"[EntryDecider] ❌ Entry GEWEIGERD voor {symbol}. Reden: {full_reason_str}. AI Intentie: {consensus_intentie}, Final AI Conf: {final_ai_confidence:.2f}, Geleerde Bias: {learned_bias:.2f} (Threshold: {entry_learned_bias_threshold:.2f}), Is Strong Pattern: {is_strong_pattern} (Weighted Score: {weighted_pattern_score:.2f} vs Threshold Param: {strong_pattern_threshold_param:.2f}).")
            return {
                "enter": False,
                "reason": full_reason_str,
                "confidence": final_ai_confidence,
                "learned_bias": learned_bias,
                "ai_intent": consensus_intentie,
                "ai_details": consensus_result,
                "pattern_details": { # Include both for clarity in logs/analysis
                    "rules": rule_patterns_data,
                    "cnn_predictions": cnn_predictions_data
                },
                "weighted_pattern_score": weighted_pattern_score
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
        original_params_get = decider.params_manager.get_param # Store original get_param
        original_cooldown_active = decider.cooldown_tracker.is_cooldown_active


        # --- Test Scenario 1: Positive Entry (Good Time of Day & Sufficient Pattern Weight) ---
        print("\n--- Test Scenario 1: Positive Entry (ML Score + Rule-based) ---")
        if decider.prompt_builder:
            async def mock_generate_prompt_positive(*args, **kwargs): return "Mock prompt for positive entry."
            decider.prompt_builder.generate_prompt_with_data = mock_generate_prompt_positive
        async def mock_ask_ai_positive(*args, **kwargs): return {"reflectie": "AI: Markt veelbelovend.", "confidence": 0.85, "intentie": "LONG", "emotie": "optimistisch", "bias": 0.7}
        decider.gpt_reflector.ask_ai = mock_ask_ai_positive
        decider.grok_reflector.ask_ai = mock_ask_ai_positive
        # Mock CNN to provide both ML score and a rule-based pattern
        async def mock_cnn_ml_and_rule(*args, **kwargs):
            # Updated structure for cnn_predictions
            return {
                "cnn_predictions": {
                    f"{mock_df_5m.attrs.get('timeframe', '5m')}_bullFlag_score": 0.95
                },
                "patterns": { # Rule-based patterns still come from 'patterns'
                    "bullishEngulfing": True
                },
                "context": {"trend": "uptrend"}
            }
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_ml_and_rule
        decider.cooldown_tracker.is_cooldown_active = lambda t, s: False

        def mock_get_param_s1(key, strategy_id=None, default=None):
            params = {
                "timeOfDayEffectiveness": {str(datetime.now().hour): 0.0}, # Neutral time effectiveness
                "cnnPatternWeight": 0.8,
                "entryLearnedBiasThreshold": 0.55, # Default
                "entryTimeEffectivenessImpactFactor": 0.5, # Default
                "entryRulePatternScore": 0.7, # Default
                "strongPatternThreshold": 0.7 # NEW: Add this for the test
            }
            return params.get(key, default) # Use default from call if key not in mock
        decider.params_manager.get_param = mock_get_param_s1

        # CALCULATION FOR SCENARIO 1 (Updated for new logic):
        # entryLearnedBiasThreshold = 0.55 (default from mock)
        # entryTimeEffectivenessImpactFactor = 0.5 (default from mock)
        # entryRulePatternScore = 0.7 (default from mock)
        # cnnPatternWeight = 0.8 (from mock_get_param_s1)
        # strongPatternThreshold = 0.7 (from mock_get_param_s1)
        # CNN score (15m_bullFlag_score) = 0.95 (from mock_cnn_ml_and_rule)
        # Rule-based pattern "bullishEngulfing" detected.
        # CNN contribution = 0.95 * 0.8 = 0.76
        # Rule contribution = 0.7 * 0.8 = 0.56
        # weighted_pattern_score = 0.76 + 0.56 = 1.32
        # is_strong_pattern = 1.32 >= strongPatternThreshold (0.7) -> True
        # AI conf from mock_ask_ai_positive = 0.85. learned_bias = 0.7 (passed to should_enter).
        # time_adjusted_confidence_multiplier = 1.0 + (0.0 * 0.5) = 1.0
        # final_ai_confidence = 0.85 * 1.0 = 0.85
        # entry_conviction_threshold = 0.7 (passed as argument to should_enter)
        # Conditions for entry:
        # consensus_intentie == "LONG" (True, from mock_ask_ai_positive)
        # final_ai_confidence (0.85) >= entry_conviction_threshold (0.7) (True)
        # learned_bias (0.7) >= entry_learned_bias_threshold (0.55) (True)
        # is_strong_pattern (True)
        # All True.
        entry_decision_s1 = await decider.should_enter(
            dataframe=mock_df_5m, symbol=test_symbol, current_strategy_id=test_strategy_id,
            trade_context=mock_trade_context,
            learned_bias=0.7, learned_confidence=0.8, entry_conviction_threshold=0.7
        )
        print(f"Resultaat (Scenario 1 - ML + Rule): {json.dumps(entry_decision_s1, indent=2, default=str)}")
        assert entry_decision_s1['enter'] is True
        assert "AI_CONSENSUS_LONG_CONDITIONS_MET" in entry_decision_s1['reason']
        assert entry_decision_s1['confidence'] == 0.85
        assert abs(entry_decision_s1['weighted_pattern_score'] - 1.32) < 0.001

        # --- Test Scenario 2: Low Weighted Pattern Score (CNN only, low score) ---
        print("\n--- Test Scenario 2: Low Weighted Pattern Score (CNN only, low score) ---")
        async def mock_cnn_low_score_only(*args, **kwargs):
            # Updated structure for cnn_predictions, no rule-based patterns
            return {
                "cnn_predictions": {
                    f"{mock_df_5m.attrs.get('timeframe', '5m')}_bullFlag_score": 0.1
                },
                "patterns": {}, # No rule-based patterns
                "context": {"trend": "neutral"}
            }
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_low_score_only
        # params_manager mock (mock_get_param_s1) is still active, giving cnnPatternWeight = 0.8
        # It also gives neutral time effect, which is fine.

        # CALCULATION FOR SCENARIO 2 (Updated for new logic):
        # params from mock_get_param_s1:
        # cnnPatternWeight = 0.8
        # strongPatternThreshold = 0.7
        # CNN score (15m_bullFlag_score) = 0.1 (from mock_cnn_low_score_only)
        # No rule-based patterns.
        # CNN contribution = 0.1 * 0.8 = 0.08
        # Rule contribution = 0
        # weighted_pattern_score = 0.08
        # is_strong_pattern = weighted_pattern_score (0.08) >= strongPatternThreshold (0.7) -> False.
        # This is the primary reason for rejection.
        entry_decision_s2 = await decider.should_enter(
            dataframe=mock_df_5m, symbol=test_symbol, current_strategy_id=test_strategy_id,
            trade_context=mock_trade_context,
            learned_bias=0.7, learned_confidence=0.8, entry_conviction_threshold=0.7
        )
        print(f"Resultaat (Scenario 2 - Low CNN Score): {json.dumps(entry_decision_s2, indent=2, default=str)}")
        assert entry_decision_s2['enter'] is False
        # The reason string includes the score vs threshold. The threshold is now strongPatternThreshold (0.7 from mock).
        assert "pattern_score_low (0.08_vs_0.70)" in entry_decision_s2['reason']


        # --- Test Scenario 3: Negative Time-of-Day Effectiveness Blocks Entry (Custom Impact Factor) ---
        print("\n--- Test Scenario 3: Negative Time-of-Day Effectiveness Blocks Entry (Custom Impact Factor) ---")
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_ml_and_rule

        def mock_get_param_s3(key, strategy_id=None, default=None):
            params = {
                "timeOfDayEffectiveness": {str(datetime.now().hour): -0.8}, # Strong negative
                "cnnPatternWeight": 0.8,
                "entryLearnedBiasThreshold": 0.55, # Default
                "entryTimeEffectivenessImpactFactor": 0.7, # Custom, higher impact
                "entryRulePatternScore": 0.7, # Default
                "strongPatternThreshold": 0.7 # Add this for the test
            }
            return params.get(key, default)
        decider.params_manager.get_param = mock_get_param_s3

        # CALCULATION FOR SCENARIO 3:
        # Params from mock_get_param_s3:
        # entryTimeEffectivenessImpactFactor = 0.7
        # cnnPatternWeight = 0.8
        # strongPatternThreshold = 0.7
        # weighted_pattern_score = 1.32 (as per mock_cnn_ml_and_rule and cnnPatternWeight=0.8 from S3 mock)
        # is_strong_pattern = 1.32 >= strongPatternThreshold (0.7) -> True.
        # AI conf 0.85 (from mock_ask_ai_positive).
        # Time mult = 1 + (-0.8 * 0.7) = 1 - 0.56 = 0.44.
        # final_ai_conf = 0.85 * 0.44 = 0.374.
        # entry_conviction_threshold = 0.7 (passed to should_enter)
        # Primary failure: final_ai_conf (0.374) < entry_conviction_threshold (0.7).
        entry_decision_s3 = await decider.should_enter(
            dataframe=mock_df_5m, symbol=test_symbol, current_strategy_id=test_strategy_id,
            trade_context=mock_trade_context,
            learned_bias=0.7, learned_confidence=0.8, entry_conviction_threshold=0.7
        )
        print(f"Resultaat (Scenario 3 - Negative Time): {json.dumps(entry_decision_s3, indent=2, default=str)}")
        assert entry_decision_s3['enter'] is False
        assert "final_ai_conf_low" in entry_decision_s3['reason']

        # --- Test Scenario 4: Cooldown Active Blocks Entry ---
        print("\n--- Test Scenario 4: Cooldown Active Blocks Entry ---")
        decider.cooldown_tracker.is_cooldown_active = lambda t, s: True # Cooldown active
        # Reset other mocks to generally positive conditions
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_ml_and_rule # uses 0.95 CNN score
        decider.params_manager.get_param = mock_get_param_s1 # cnnPatternWeight=0.8, neutral time

        entry_decision_s4 = await decider.should_enter(
            dataframe=mock_df_5m, symbol=test_symbol, current_strategy_id=test_strategy_id,
            trade_context=mock_trade_context,
            learned_bias=0.7, learned_confidence=0.8, entry_conviction_threshold=0.7
        )
        print(f"Resultaat (Scenario 4 - Cooldown): {json.dumps(entry_decision_s4, indent=2, default=str)}")
        assert entry_decision_s4['enter'] is False
        assert "ai_cooldown_active" in entry_decision_s4['reason']


        # Restore original methods
        if decider.prompt_builder:
            decider.prompt_builder.generate_prompt_with_data = original_prompt_builder_generate
        decider.gpt_reflector.ask_ai = original_gpt_ask
        decider.grok_reflector.ask_grok = original_grok_ask
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = original_cnn_detect
        decider.bias_reflector.get_bias_score = original_bias_get
        if decider.confidence_engine:
            decider.confidence_engine.get_confidence_score = original_conf_get
        decider.params_manager.get_param = original_params_get # Restore original get_param
        decider.cooldown_tracker.is_cooldown_active = original_cooldown_active


    asyncio.run(run_test_entry_decider())
