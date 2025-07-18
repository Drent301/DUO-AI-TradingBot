# core/exit_optimizer.py
import logging
from typing import Dict, Any, Optional
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import dotenv
import re
import json

# Importeer AI-modules
from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.cnn_patterns import CNNPatterns
from core.params_manager import ParamsManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ExitOptimizer:
    """
    Biedt AI-gestuurde logica voor exitbesluiten, inclusief dynamische aanpassing
    van Freqtrade's trailing stop en dynamic stop loss.
    """

    def __init__(self):
        self.gpt_reflector = GPTReflector()
        self.grok_reflector = GrokReflector()
        try:
            self.prompt_builder = PromptBuilder()
        except Exception as e:
            logger.error(f"Failed to initialize PromptBuilder: {e}", exc_info=True)
            self.prompt_builder = None
        self.bias_reflector = BiasReflector()
        try:
            self.confidence_engine = ConfidenceEngine()
        except Exception as e:
            logger.error(f"Failed to initialize ConfidenceEngine: {e}", exc_info=True)
            self.confidence_engine = None

        self.cnn_patterns_detector = CNNPatterns()
        self.params_manager = ParamsManager() # Instantie van ParamsManager
        logger.info("ExitOptimizer geïnitialiseerd.")

    async def should_exit(
        self,
        dataframe: pd.DataFrame,
        trade: Dict[str, Any], # Freqtrade trade object of vergelijkbaar
        symbol: str,
        current_strategy_id: str,
        # De volgende zijn leerbare parameters, doorgegeven vanuit de strategie
        learned_bias: float = 0.5,
        learned_confidence: float = 0.5,
        exit_conviction_drop_trigger: float = 0.4, # Default if not passed
        candles_by_timeframe: Optional[Dict[str, pd.DataFrame]] = None, # Relevant voor AI, made Optional with default
        additional_context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Bepaalt of een exit moet worden geplaatst op basis van AI-consensus en drempelwaarden.
        """
        logger.debug(f"[ExitOptimizer] Evalueren exit voor {symbol} (trade_id: {trade.get('id', 'N/A')})...")

        if dataframe.empty:
            logger.warning(f"[ExitOptimizer] Geen dataframe beschikbaar voor {symbol}. Kan geen exit besluit nemen.")
            return {"exit": False, "reason": "no_dataframe", "confidence": 0.0, "pattern_details": {}}

        # Ensure candles_by_timeframe is not None and contains data
        if candles_by_timeframe is None: # Check if None was explicitly passed or default is used
            candles_by_timeframe = {dataframe.attrs.get('timeframe', '5m'): dataframe.copy()}
            logger.debug(f"Using only current timeframe DF for exit_optimizer, as no multi-timeframe dict was provided.")


        # Genereer prompt voor AI
        prompt = None
        if self.prompt_builder:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe, # Nu correct doorgegeven
                symbol=symbol,
                prompt_type='riskManagement', # Or 'marketAnalysis' for exit
                current_bias=learned_bias,
                current_confidence=learned_confidence,
                additional_context=additional_context_data # Pass the new parameter
            )

        if not prompt:
            logger.warning(f"[ExitOptimizer] Geen prompt gegenereerd voor {symbol}. Exit evaluatie kan minder accuraat zijn.")
            # Fallback prompt generation remains if prompt is None
            prompt = f"Analyseer de huidige markt voor {symbol} en de openstaande trade ({trade.get('id','N/A')}) voor een mogelijk exit signaal. Huidige profit: {trade.get('profit_pct',0):.2%}"
        else:
            # New log statement
            sentiment_status = "absent"
            if additional_context_data and 'social_sentiment_grok' in additional_context_data:
                if additional_context_data['social_sentiment_grok']: # Check if list is not empty
                    sentiment_status = "present and not empty"
                else:
                    sentiment_status = "present but empty"

            logger.info(f"[ExitOptimizer] AI prompt for {symbol} (should_exit) generated. Social sentiment data was {sentiment_status} in the prompt context.")

        # Vraag AI-consensus
        gpt_response = await self.gpt_reflector.ask_ai(prompt, context={"symbol": symbol, "strategy_id": current_strategy_id, "trade": trade})
        grok_response = await self.grok_reflector.ask_grok(prompt, context={"symbol": symbol, "strategy_id": current_strategy_id, "trade": trade})

        gpt_confidence = gpt_response.get('confidence', 0.0) or 0.0
        grok_confidence = grok_response.get('confidence', 0.0) or 0.0
        gpt_intentie = str(gpt_response.get('intentie', '')).upper()
        grok_intentie = str(grok_response.get('intentie', '')).upper()

        # Enhanced logging for individual AI responses in should_exit
        logger.info(f"GPT Raw Reflectie for {symbol} (should_exit, first 200 chars): {gpt_response.get('reflectie', '')[:200]}")
        logger.info(f"Grok Raw Reflectie for {symbol} (should_exit, first 200 chars): {grok_response.get('reflectie', '')[:200]}")
        logger.info(f"GPT Parsed for {symbol} (should_exit): Intentie='{gpt_intentie}', Confidence={gpt_confidence:.2f}")
        logger.info(f"Grok Parsed for {symbol} (should_exit): Intentie='{grok_intentie}', Confidence={grok_confidence:.2f}")

        num_valid_confidences = sum(1 for conf in [gpt_confidence, grok_confidence] if isinstance(conf, (float, int)) and conf >= 0)
        if num_valid_confidences > 0:
            combined_confidence = (gpt_confidence + grok_confidence) / num_valid_confidences
        else:
            combined_confidence = 0.0

        ai_exit_intent = False
        if (gpt_intentie == "SELL" and grok_intentie == "SELL") or \
           (gpt_intentie == "SELL" and grok_intentie in ["HOLD", ""] and gpt_confidence > 0.6) or \
           (grok_intentie == "SELL" and gpt_intentie in ["HOLD", ""] and grok_confidence > 0.6):
            ai_exit_intent = True


        # Check voor bearish patronen (uit cnn_patterns.py)
        pattern_data = await self.cnn_patterns_detector.detect_patterns_multi_timeframe(candles_by_timeframe, symbol)

        # 1. Fetch Parameters
        cnn_pattern_weight = self.params_manager.get_param("cnnPatternWeight", strategy_id=current_strategy_id, default=1.0)
        if not isinstance(cnn_pattern_weight, (float, int)): # Robust handling
            logger.warning(f"Invalid cnnPatternWeight type ({type(cnn_pattern_weight)}) for strategy {current_strategy_id}, using default 1.0.")
            cnn_pattern_weight = 1.0
        logger.info(f"Symbol: {symbol} (Exit), Using cnnPatternWeight: {cnn_pattern_weight} (Default: 1.0)")

        exit_rule_pattern_score_param = self.params_manager.get_param("exitRulePatternScore", strategy_id=current_strategy_id, default=0.7)
        logger.info(f"Symbol: {symbol} (Exit), Using exitRulePatternScore: {exit_rule_pattern_score_param} (Default: 0.7)")

        strong_pattern_threshold_param = self.params_manager.get_param(
            "strongPatternThreshold", # Changed key
            strategy_id=current_strategy_id,
            default=0.5
        )
        logger.info(f"Symbol: {symbol} (Exit), Using strongPatternThreshold: {strong_pattern_threshold_param} (Default: 0.5)")

        # 2. Calculate weighted_bearish_pattern_score
        weighted_bearish_pattern_score = 0.0
        detected_patterns_summary = []
        cnn_predictions_data = pattern_data.get('cnn_predictions', {})
        rule_patterns_data = pattern_data.get('patterns', {})

        # CNN Numerical Prediction Score
        current_timeframe = dataframe.attrs.get('timeframe', '5m')
        # Define target bearish CNN score keys - Changed to no_bullFlag_score
        target_bearish_cnn_keys = [f"{current_timeframe}_no_bullFlag_score"]
        # Consider adding others like _darkCloudCover_score, _shootingStar_score if conventional and available

        if cnn_predictions_data: # Check if cnn_predictions dictionary exists and is not empty
            logger.info(f"Symbol: {symbol} (Exit), Evaluating CNN predictions: {cnn_predictions_data} for keys: {target_bearish_cnn_keys}")
            for key in target_bearish_cnn_keys:
                cnn_score = cnn_predictions_data.get(key)
                if cnn_score is not None and isinstance(cnn_score, (float, int)) and cnn_score > 0:
                    contribution = cnn_score * cnn_pattern_weight
                    weighted_bearish_pattern_score += contribution
                    logger.info(f"Symbol: {symbol} (Exit), CNN no_bullFlag_score (as bearish signal) contribution from '{key}': {cnn_score:.4f} * {cnn_pattern_weight:.2f} = {contribution:.4f}. Current weighted_score: {weighted_bearish_pattern_score:.4f}")
                    detected_patterns_summary.append(f"CNN_{key}({cnn_score:.4f})")
                    break # Break after the first positive CNN score contribution
        else:
            logger.info(f"Symbol: {symbol} (Exit), No CNN predictions found in pattern_data.")

        # Rule-Based Pattern Score
        rule_based_bearish_patterns = [
            'bearishEngulfing', 'CDLENGULFING', 'eveningStar', 'CDLEVENINGSTAR',
            'threeBlackCrows', 'CDL3BLACKCROWS', 'darkCloudCover', 'CDLDARKCLOUDCOVER',
            'bearishRSIDivergence', 'CDLHANGINGMAN', 'doubleTop', 'descendingTriangle', 'parabolicCurveDown'
        ]

        if isinstance(rule_patterns_data, dict):
            logger.info(f"Symbol: {symbol} (Exit), Evaluating rule-based patterns: {rule_patterns_data}")
            for p_name in rule_based_bearish_patterns:
                if rule_patterns_data.get(p_name, False): # Checks for existence and truthiness
                    contribution = exit_rule_pattern_score_param * cnn_pattern_weight # Use fetched exitRulePatternScore
                    weighted_bearish_pattern_score += contribution
                    logger.info(f"Symbol: {symbol} (Exit), Detected rule-based bearish pattern: '{p_name}'. Added contribution: {exit_rule_pattern_score_param:.2f} * {cnn_pattern_weight:.2f} = {contribution:.4f}. Current weighted_score: {weighted_bearish_pattern_score:.4f}")
                    detected_patterns_summary.append(f"rule({p_name})")
                    break # Only the first detected rule-based bearish pattern contributes
        else:
            logger.warning(f"Symbol: {symbol} (Exit), 'patterns' key missing or not a dict in pattern_data. Skipping rule-based pattern check.")

        if not detected_patterns_summary: # Simplified check
             logger.info(f"Symbol: {symbol} (Exit), No contributing CNN or rule-based bearish patterns detected.")

        logger.info(f"Symbol: {symbol} (Exit), Final Calculated Weighted Bearish Pattern Score: {weighted_bearish_pattern_score:.4f} from patterns: [{'; '.join(detected_patterns_summary)}]")

        # 3. Determine is_strong_bearish_pattern
        is_strong_bearish_pattern = weighted_bearish_pattern_score >= strong_pattern_threshold_param # Use renamed variable

        log_msg_pattern_strength = (f"Strong bearish pattern {'DETECTED' if is_strong_bearish_pattern else 'NOT detected'}. "
                                    f"Score {weighted_bearish_pattern_score:.4f} "
                                    f"{'>=' if is_strong_bearish_pattern else '<'} Threshold {strong_pattern_threshold_param:.2f}") # Use renamed variable
        logger.info(f"Symbol: {symbol} (Exit), {log_msg_pattern_strength}")

        # AI-besluitvormingslogica voor exit
        current_profit_pct = trade.get('profit_pct', 0.0)
        # exit_conviction_drop_trigger is passed as argument, or use default from signature

        # Scenario 1: AI is onzeker en trade is in winst (take profit)
        # exit_conviction_drop_trigger is an argument to should_exit, so already configurable.
        # The 0.005 (0.5%) profit condition can also be made configurable if needed.
        min_profit_for_low_conf_exit = self.params_manager.get_param("minProfitForLowConfExit", strategy_id=current_strategy_id, default=0.005)
        logger.info(f"Symbol: {symbol} (Exit), Using minProfitForLowConfExit: {min_profit_for_low_conf_exit} (Default: 0.005)")

        if combined_confidence < exit_conviction_drop_trigger and current_profit_pct > min_profit_for_low_conf_exit:
            logger.info(f"[ExitOptimizer] ✅ Exit door lage AI confidence ({combined_confidence:.2f} < {exit_conviction_drop_trigger}) en trade in winst ({current_profit_pct:.2%}) voor {symbol}.")
            return {"exit": True, "reason": "low_ai_confidence_profit_taking", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {}), "weighted_bearish_score": weighted_pattern_score}

        # Fetch confidence thresholds for exit decisions from ParamsManager
        exit_pattern_conf_threshold = self.params_manager.get_param("exitPatternConfThreshold", strategy_id=current_strategy_id, default=0.5)
        exit_ai_sell_intent_conf_threshold = self.params_manager.get_param("exitAISellIntentConfThreshold", strategy_id=current_strategy_id, default=0.6)
        logger.info(f"Symbol: {symbol} (Exit), Using exitPatternConfThreshold: {exit_pattern_conf_threshold} (Default: 0.5)")
        logger.info(f"Symbol: {symbol} (Exit), Using exitAISellIntentConfThreshold: {exit_ai_sell_intent_conf_threshold} (Default: 0.6)")

        # Scenario 2: Sterk bearish patroon met AI-bevestiging
        if is_strong_bearish_pattern and combined_confidence > exit_pattern_conf_threshold:
             logger.info(f"[ExitOptimizer] ✅ Exit door {log_msg_pattern_strength} en AI confidence {combined_confidence:.2f} (>{exit_pattern_conf_threshold:.2f}) voor {symbol}.")
             return {"exit": True, "reason": "bearish_pattern_with_ai_confirmation", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {}), "weighted_bearish_score": weighted_pattern_score}

        # Scenario 3: AI wil verkopen met voldoende confidence
        if ai_exit_intent and combined_confidence > exit_ai_sell_intent_conf_threshold:
            logger.info(f"[ExitOptimizer] ✅ Exit door AI verkoop intentie (GPT: {gpt_intentie}, Grok: {grok_intentie}) met confidence {combined_confidence:.2f} (>{exit_ai_sell_intent_conf_threshold:.2f}) voor {symbol}.")
            return {"exit": True, "reason": "ai_sell_intent_confident", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {}), "weighted_bearish_score": weighted_pattern_score}

        logger.debug(f"[ExitOptimizer] Geen AI-gedreven exit trigger voor {symbol}. AI Conf: {combined_confidence:.2f}, AI Intent GPT: {gpt_intentie}, Grok: {grok_intentie}, Weighted Bearish Score: {weighted_pattern_score:.2f}.")
        return {"exit": False, "reason": "no_ai_exit_signal", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {}), "weighted_bearish_score": weighted_pattern_score}

    async def optimize_trailing_stop_loss(
        self,
        dataframe: pd.DataFrame,
        trade: Dict[str, Any],
        symbol: str,
        current_strategy_id: str,
        # Doorsturen van geleerde parameters en candles_by_timeframe
        learned_bias: float = 0.5,
        learned_confidence: float = 0.5,
        candles_by_timeframe: Optional[Dict[str, pd.DataFrame]] = None, # Made Optional with default
        additional_context_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, float]]:
        """
        Optimaliseert dynamisch de trailing stop loss op basis van AI-inzichten.
        """
        logger.debug(f"[ExitOptimizer] Optimaliseren trailing stop loss voor {symbol} (trade_id: {trade.get('id', 'N/A')})...")

        if candles_by_timeframe is None: # Check if None was explicitly passed or default is used
            candles_by_timeframe = {dataframe.attrs.get('timeframe', '5m'): dataframe.copy()}
            logger.debug(f"Using only current timeframe DF for TSL optimization, as no multi-timeframe dict was provided.")


        prompt = None
        if self.prompt_builder:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe,
                symbol=symbol,
                prompt_type='riskManagement',
                current_bias=learned_bias,
                current_confidence=learned_confidence,
                additional_context=additional_context_data # Pass the new parameter
            )
        if not prompt:
            logger.warning(f"[ExitOptimizer] Geen risicomanagement prompt gegenereerd voor {symbol}. SL niet geoptimaliseerd.")
            return None
        else:
            # New log statement
            sentiment_status = "absent"
            if additional_context_data and 'social_sentiment_grok' in additional_context_data:
                if additional_context_data['social_sentiment_grok']: # Check if list is not empty
                    sentiment_status = "present and not empty"
                else:
                    sentiment_status = "present but empty"

            logger.info(f"[ExitOptimizer] AI prompt for {symbol} (optimize_trailing_stop_loss) generated. Social sentiment data was {sentiment_status} in the prompt context.")

        # Vraag AI om SL-advies
        gpt_response = await self.gpt_reflector.ask_ai(prompt, context={"symbol": symbol, "strategy_id": current_strategy_id, "trade": trade})
        grok_response = await self.grok_reflector.ask_grok(prompt, context={"symbol": symbol, "strategy_id": current_strategy_id, "trade": trade})

        # Enhanced logging for individual AI responses in optimize_trailing_stop_loss
        logger.info(f"GPT Raw Reflectie for {symbol} (optimize_SL, first 200 chars): {gpt_response.get('reflectie', '')[:200]}")
        logger.info(f"Grok Raw Reflectie for {symbol} (optimize_SL, first 200 chars): {grok_response.get('reflectie', '')[:200]}")

        ai_recommended_sl_pct: Optional[float] = None
        highest_confidence_for_sl = 0.0
        sl_keyword_mentioned_in_any_response = False

        # Regex to find the specific JSON object for recommended_sl_percentage
        json_object_pattern = re.compile(r"(\{\s*\"recommended_sl_percentage\"\s*:\s*\d+(?:\.\d+)?\s*\})")

        # Regex patterns for fallback (existing logic)
        sl_regex_patterns = [
            re.compile(r"(?:stop(?:[-_\s]?loss)?|s(?:[-_\s]?l)?|trailing(?:[-_\s]?stop)?|tsl)\s*(?:at|is|to|of|should be|[:])?\s*(\d+(?:\.\d+)?)\s*(?:%|pct|percent)\b", re.IGNORECASE),
            re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|pct|percent)\s*(?:stop(?:[-_\s]?loss)?|s(?:[-_\s]?l)?|trailing(?:[-_\s]?stop)?|tsl)\b", re.IGNORECASE),
            re.compile(r"(?:stop(?:[-_\s]?loss)?|s(?:[-_\s]?l)?|trailing(?:[-_\s]?stop)?|tsl)\s*.*?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
        ]
        sl_keyword_regex = re.compile(r"\b(?:stop(?:[-_\s]?loss)?|s(?:[-_\s]?l)?|trailing(?:[-_\s]?stop)?|tsl)\b", re.IGNORECASE)

        for i, resp in enumerate([gpt_response, grok_response]):
            ai_name = "GPT" if i == 0 else "Grok"
            reflection = resp.get('reflectie', '')
            resp_confidence = resp.get('confidence', 0.0) or 0.0
            parsed_sl_value_for_this_response = None
            sl_parsed_via_json = False

            if sl_keyword_regex.search(reflection): # Check if any SL keyword is mentioned, for generic logging later
                sl_keyword_mentioned_in_any_response = True

            if resp_confidence <= 0.55: # Threshold for considering SL advice at all
                logger.debug(f"[{ai_name}] AI response confidence {resp_confidence:.2f} too low for SL parsing. Skipping.")
                continue

            # 1. Attempt JSON parsing first
            json_match = json_object_pattern.search(reflection)
            if json_match:
                json_str = json_match.group(1)
                try:
                    sl_data = json.loads(json_str)
                    if "recommended_sl_percentage" in sl_data:
                        sl_value_raw = sl_data["recommended_sl_percentage"]
                        if isinstance(sl_value_raw, (float, int)):
                            sl_value_float = float(sl_value_raw) / 100.0  # Convert X.Y to 0.0X Y

                            if 0.001 <= sl_value_float <= 0.50:  # 0.1% to 50%
                                logger.info(f"[{ai_name}] Successfully parsed SL via JSON: {sl_value_float:.2%} (raw: {sl_value_raw}). Confidence: {resp_confidence:.2f}.")
                                if resp_confidence > highest_confidence_for_sl:
                                    parsed_sl_value_for_this_response = sl_value_float
                                    highest_confidence_for_sl = resp_confidence
                                    sl_parsed_via_json = True # Mark as successfully parsed by JSON
                                else:
                                    logger.info(f"[{ai_name}] JSON SL value {sl_value_float:.2%} not used due to lower confidence than current best ({highest_confidence_for_sl:.2f}).")
                            else:
                                logger.warning(f"[{ai_name}] JSON SL value {sl_value_float:.2%} (raw: {sl_value_raw}) is outside reasonable range (0.1%-50%). Not used. Reflection: '{reflection}'")
                        else:
                            logger.warning(f"[{ai_name}] JSON 'recommended_sl_percentage' value '{sl_value_raw}' is not a valid number. Reflection: '{reflection}'")
                    else:
                        logger.warning(f"[{ai_name}] Found JSON object but 'recommended_sl_percentage' key is missing. JSON: '{json_str}'. Reflection: '{reflection}'")
                except json.JSONDecodeError as e:
                    logger.warning(f"[{ai_name}] Failed to decode JSON string '{json_str}' for SL parsing: {e}. Reflection: '{reflection}'")
            else:
                logger.info(f"[{ai_name}] No specific SL JSON object found in reflection. Will attempt regex parsing.")

            # 2. Fallback to Regex Parsing if JSON parsing was not successful for this response
            if not sl_parsed_via_json:
                logger.info(f"[{ai_name}] Attempting regex fallback for SL parsing.")
                for pattern_idx, sl_regex in enumerate(sl_regex_patterns):
                    sl_match = sl_regex.search(reflection)
                    if sl_match:
                        try:
                            sl_value_str = sl_match.group(1)
                            sl_value_float = float(sl_value_str) / 100.0

                            if 0.001 <= sl_value_float <= 0.50:
                                logger.info(f"[{ai_name}] Parsed SL via REGEX: {sl_value_float:.2%} from pattern #{pattern_idx+1}. Confidence: {resp_confidence:.2f}.")
                                if resp_confidence > highest_confidence_for_sl:
                                    parsed_sl_value_for_this_response = sl_value_float
                                    highest_confidence_for_sl = resp_confidence
                                    # No sl_parsed_via_json = True here
                                else:
                                     logger.info(f"[{ai_name}] Regex SL value {sl_value_float:.2%} not used due to lower confidence than current best ({highest_confidence_for_sl:.2f}).")
                                break # Found a match with this regex, no need to try others for this response
                            else:
                                logger.warning(f"[{ai_name}] Regex parsed SL value {sl_value_float:.2%} is outside reasonable range. Ignored. Reflection: '{reflection}'")
                        except ValueError:
                            logger.warning(f"[{ai_name}] Could not parse SL percentage (ValueError) from regex match '{sl_match.group(1)}' in reflection: '{reflection}'")
                        except Exception as e:
                            logger.error(f"[{ai_name}] Unexpected error parsing SL from regex match '{sl_match.group(1)}': {e}. Reflection: '{reflection}'", exc_info=True)

            if parsed_sl_value_for_this_response is not None and (resp_confidence == highest_confidence_for_sl):
                # If this response yields a valid SL and has the highest confidence so far (or equal, choosing the latest one)
                ai_recommended_sl_pct = parsed_sl_value_for_this_response
                logger.info(f"[{ai_name}] Setting AI recommended SL to {ai_recommended_sl_pct:.2%} based on this response (Confidence: {resp_confidence:.2f}, Parsed via JSON: {sl_parsed_via_json}).")


        if ai_recommended_sl_pct is None:
            if sl_keyword_mentioned_in_any_response:
                logger.warning(f"[ExitOptimizer] AI mentioned stop-loss keywords but a specific percentage could not be parsed reliably (JSON or Regex) from responses for {symbol}.")
            else:
                logger.info(f"[ExitOptimizer] No AI advice for SL optimization (no SL percentage parsed, or no keywords mentioned) for {symbol}.")
            logger.warning(f"[ExitOptimizer] AI mentioned stop-loss keywords but a specific percentage could not be parsed reliably (JSON or Regex) from responses for {symbol}.")
            else:
                logger.info(f"[ExitOptimizer] No AI advice for SL optimization (no SL percentage parsed, or no keywords mentioned) for {symbol}.")
            return None

        # If ai_recommended_sl_pct has a value, proceed with logic
        logger.info(f"[ExitOptimizer] Highest confidence AI-recommended SL: {ai_recommended_sl_pct:.2%} (final confidence for this SL: {highest_confidence_for_sl:.2f}) for {symbol}.")

        # Fetch SL adjustment parameters
        sl_bias_impact_factor = self.params_manager.get_param("slBiasImpactFactor", strategy_id=current_strategy_id, default=0.2)
        sl_confidence_impact_factor = self.params_manager.get_param("slConfidenceImpactFactor", strategy_id=current_strategy_id, default=0.2)
        sl_min_offset = self.params_manager.get_param("slMinOffset", strategy_id=current_strategy_id, default=0.005)
        sl_max_offset = self.params_manager.get_param("slMaxOffset", strategy_id=current_strategy_id, default=0.10)
        sl_hard_stop_multiplier = self.params_manager.get_param("slHardStopMultiplier", strategy_id=current_strategy_id, default=1.5) # Multiplier for hard stop based on offset
        sl_min_hard_stop = self.params_manager.get_param("slMinHardStop", strategy_id=current_strategy_id, default=-0.20)
        sl_max_hard_stop = self.params_manager.get_param("slMaxHardStop", strategy_id=current_strategy_id, default=-0.01)
        sl_default_trigger_profit = self.params_manager.get_param("slDefaultTriggerProfit", strategy_id=current_strategy_id, default=0.005)
        sl_trigger_profit_threshold = self.params_manager.get_param("slTriggerProfitThreshold", strategy_id=current_strategy_id, default=0.05) # Profit level to adjust trigger
        sl_trigger_profit_factor = self.params_manager.get_param("slTriggerProfitFactor", strategy_id=current_strategy_id, default=0.5) # Factor to adjust trigger by

        logger.info(f"[{symbol}] SL Opt. Params: biasImpact={sl_bias_impact_factor}, confImpact={sl_confidence_impact_factor}, minOffset={sl_min_offset}, maxOffset={sl_max_offset}, "
                    f"hardStopMultiplier={sl_hard_stop_multiplier}, minHardStop={sl_min_hard_stop}, maxHardStop={sl_max_hard_stop}, defaultTrigger={sl_default_trigger_profit}, "
                    f"triggerProfitThreshold={sl_trigger_profit_threshold}, triggerProfitFactor={sl_trigger_profit_factor}")

        adjusted_trailing_offset = ai_recommended_sl_pct # Start with AI recommendation

            # Adjust based on bias and confidence
            bias_factor = 1.0 - ((learned_bias - 0.5) * sl_bias_impact_factor)
            confidence_factor = 1.0 - ((learned_confidence - 0.5) * sl_confidence_impact_factor)

            final_trailing_offset = adjusted_trailing_offset * bias_factor * confidence_factor
            final_trailing_offset = max(sl_min_offset, min(final_trailing_offset, sl_max_offset))

            # Calculate a hard stoploss based on this
            hard_stoploss_value = -(final_trailing_offset * sl_hard_stop_multiplier)
            hard_stoploss_value = max(sl_min_hard_stop, min(hard_stoploss_value, sl_max_hard_stop))

            # Dynamic trailing stop trigger based on profit
            trailing_stop_trigger = sl_default_trigger_profit
            current_profit_pct = trade.get('profit_pct', 0.0)
            if current_profit_pct > sl_trigger_profit_threshold:
                trailing_stop_trigger = current_profit_pct * sl_trigger_profit_factor

            logger.info(f"[ExitOptimizer] AI SL Optimalisatie voor {symbol}: "
                        f"Hard SL: {hard_stoploss_value:.2%}, "
                        f"Trailing Trigger: {trailing_stop_trigger:.2%}, "
                        f"Trailing Offset: {final_trailing_offset:.2%}")

            return {
                "stoploss": hard_stoploss_value,
                "trailing_stop_positive_offset": final_trailing_offset,
                "trailing_stop_positive": trailing_stop_trigger
            }

        # logger.debug(f"[ExitOptimizer] Geen AI-advies voor SL-optimalisatie voor {symbol} of confidence te laag.") # Replaced by more specific logs above
        return None

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    # Use pathlib for robust path handling
    from pathlib import Path
    dotenv_path = Path(__file__).resolve().parent.parent / '.env'
    dotenv.load_dotenv(dotenv_path)

    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


    def create_mock_dataframe_for_exit_optimizer(timeframe: str = '5m', num_candles: int = 100) -> pd.DataFrame:
        data = []
        now = datetime.utcnow() # Changed from now() to utcnow() for consistency
        interval_seconds_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600} # Added '15m'
        interval_seconds = interval_seconds_map.get(timeframe, 300)

        for i in range(num_candles):
            date = now - timedelta(seconds=(num_candles - 1 - i) * interval_seconds)
            open_ = 100 + i * 0.1 + np.random.rand() * 2
            close_ = open_ + (np.random.rand() - 0.5) * 5
            high_ = max(open_, close_) + np.random.rand() * 2
            low_ = min(open_, close_) - np.random.rand() * 2
            volume = 1000 + np.random.rand() * 500
            rsi = 50 + (np.random.rand() - 0.5) * 30 # Simplified RSI
            macd_val = (np.random.rand() - 0.5) * 0.1
            macdsignal_val = macd_val * (0.8 + np.random.rand()*0.2)
            macdhist_val = macd_val - macdsignal_val
            # Simplified Bollinger Bands logic for mock data
            sma_period = 20; std_dev_multiplier = 2
            if i >= sma_period-1 and len(data) >= sma_period-1: # Ensure enough data for rolling calculation
                # Use close prices from already appended data for realistic SMA
                recent_closes_for_bb = [r[4] for r in data[-(sma_period-1):]] + [close_]
                if len(recent_closes_for_bb) == sma_period: # Check if we have exactly sma_period points
                    sma = np.mean(recent_closes_for_bb); std_dev = np.std(recent_closes_for_bb)
                    bb_middle = sma; bb_upper = sma + std_dev_multiplier * std_dev; bb_lower = sma - std_dev_multiplier * std_dev
                else: # Fallback if not enough data yet for full SMA
                    bb_middle=open_; bb_upper=high_; bb_lower=low_
            else: # Fallback for initial candles
                bb_middle=open_; bb_upper=high_; bb_lower=low_
            data.append([date,open_,high_,low_,close_,volume,rsi,macd_val,macdsignal_val,macdhist_val,bb_upper,bb_middle,bb_lower])

        df = pd.DataFrame(data, columns=['date','open','high','low','close','volume','rsi','macd','macdsignal','macdhist','bb_upperband','bb_middleband','bb_lowerband'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.attrs['timeframe'] = timeframe
        df.attrs['pair'] = 'ETH/USDT' # Added pair attribute
        return df

    async def run_test_exit_optimizer():
        optimizer = ExitOptimizer()
        test_symbol = "ETH/USDT"
        test_strategy_id = "DUOAI_Strategy"
        mock_trade_profitable = {"id": "test_trade_1", "pair": test_symbol, "is_open": True, "profit_pct": 0.03, "stake_amount": 100, "open_date": datetime.utcnow() - timedelta(hours=2)}
        mock_trade_losing = {"id": "test_trade_2", "pair": test_symbol, "is_open": True, "profit_pct": -0.02, "stake_amount": 100, "open_date": datetime.utcnow() - timedelta(hours=1)}


        mock_df = create_mock_dataframe_for_exit_optimizer(timeframe='5m', num_candles=60)
        # Mock bearish patroon voor exit test
        last_idx = mock_df.index[-1]
        # Make last candle more bearish for pattern detection test
        mock_df.loc[last_idx, 'open'] = mock_df.loc[last_idx, 'high'] - 0.1
        mock_df.loc[last_idx, 'close'] = mock_df.loc[last_idx, 'low'] + 0.1
        mock_df.loc[last_idx, 'volume'] = mock_df['volume'].mean() * 4

        mock_candles_by_timeframe = { # Renamed from mock_candles_by_timeframe_for_ai
            '5m': mock_df,
            '1h': create_mock_dataframe_for_exit_optimizer('1h', 60)
        }

        # Store original methods
        original_prompt_builder_generate = optimizer.prompt_builder.generate_prompt_with_data if optimizer.prompt_builder else None
        original_gpt_ask = optimizer.gpt_reflector.ask_ai
        original_grok_ask = optimizer.grok_reflector.ask_grok
        original_cnn_detect = optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe
        original_bias_get = optimizer.bias_reflector.get_bias_score
        original_conf_get = optimizer.confidence_engine.get_confidence_score if optimizer.confidence_engine else None
        original_params_get = optimizer.params_manager.get_param


        # Mock dependencies
        if optimizer.prompt_builder:
            # Ensure mock returns awaitable if original is async
            async def mock_generate_prompt_with_data(*args, **kwargs):
                # Simulate async behavior; the original function might not actually sleep.
                # If generate_prompt_with_data isn't truly async, this await is not needed.
                # await asyncio.sleep(0.01)
                return f"Mock prompt for {kwargs.get('symbol','N/A')} ({kwargs.get('prompt_type','N/A')})"
            optimizer.prompt_builder.generate_prompt_with_data = mock_generate_prompt_with_data

        async def mock_ask_ai(*args, **kwargs): # For gpt_reflector.ask_ai
            # await asyncio.sleep(0.01) # Not needed if the original isn't sleeping
            # Default mock response, can be overridden per test case
            return {"reflectie": "Default AI reflectie.", "confidence": 0.5, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai
        optimizer.grok_reflector.ask_grok = mock_ask_ai # Use same mock for grok for simplicity

        async def mock_detect_patterns(*args, **kwargs): # For cnn_patterns_detector
            # await asyncio.sleep(0.01) # Not needed if the original isn't sleeping
            # Default: no rule-based patterns, no CNN predictions
            return {"patterns": {}, "cnn_predictions": {}}
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_patterns

        optimizer.bias_reflector.get_bias_score = lambda t, s: 0.4
        if optimizer.confidence_engine:
            optimizer.confidence_engine.get_confidence_score = lambda t, s: 0.6

        # Centralized mock for params_manager.get_param
        def mock_params_get_for_tests(key, strategy_id=None, default=None):
            params = {
                "cnnPatternWeight": 0.8,
                "exitRulePatternScore": 0.7,
                "strongPatternThreshold": 0.75, # Changed from strongBearishPatternThreshold
                "exitConvictionDropTrigger": 0.4, # Default for this test suite
                "minProfitForLowConfExit": 0.005, # Default
                "exitPatternConfThreshold": 0.5, # Default
                "exitAISellIntentConfThreshold": 0.6, # Default
                # SL related params, keep defaults or specify if needed for SL tests
                "slBiasImpactFactor": 0.2,
                "slConfidenceImpactFactor": 0.2,
                "slMinOffset": 0.005,
                "slMaxOffset": 0.10,
                "slHardStopMultiplier": 1.5,
                "slMinHardStop": -0.20,
                "slMaxHardStop": -0.01,
                "slDefaultTriggerProfit": 0.005,
                "slTriggerProfitThreshold": 0.05,
                "slTriggerProfitFactor": 0.5,
            }
            return params.get(key, default)
        optimizer.params_manager.get_param = mock_params_get_for_tests

        # Centralized mock for cnn_patterns_detector.detect_patterns_multi_timeframe
        mock_df_timeframe = mock_df.attrs.get('timeframe', '5m')
        default_cnn_mock_data = {
            "patterns": {"bearishEngulfing": True, "eveningStar": False}, # bearishEngulfing is the first rule pattern
            "cnn_predictions": {
                f"{mock_df_timeframe}_no_bullFlag_score": 0.9, # Changed key, high bearish signal
                f"{mock_df_timeframe}_bullFlag_score": 0.1    # Optional bullFlag_score
            }
        }
        # Use deepcopy if the mock data is mutable and modified by tests, not strictly needed here as returning new dict
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = lambda *args, **kwargs: default_cnn_mock_data.copy()

        mock_additional_context_for_exit = {
            "current_trade_duration_hours": 2.5, # Example other data
            "market_volatility_index": 0.7,    # Example other data
            "social_sentiment_grok": [
                {"source": "ExitFeed", "text": "Market looks shaky for ETH.", "sentiment": "negative"},
                {"source": "NewsSite", "text": "Neutral to bearish sentiment overall.", "sentiment": "neutral"}
            ]
        }

        # --- Test should_exit ---
        print("\n--- Test ExitOptimizer (should_exit) ---")

        # Scenario: Strong Bearish Pattern (using centralized mocks) - EXIT
        # This scenario was previously named "Scenario 1A"
        print("\n--- Test Scenario: Strong Bearish Pattern (Centralized Mocks) - EXIT ---")
        # Params from mock_params_get_for_tests:
        # cnnPatternWeight = 0.8
        # exitRulePatternScore = 0.7
        # strongPatternThreshold = 0.75
        # CNN data from default_cnn_mock_data:
        # CNN score for f"{mock_df_timeframe}_no_bullFlag_score" = 0.9
        # Rule pattern = "bearishEngulfing": True
        # Calculation:
        # CNN contribution (no_bullFlag_score) = 0.9 * 0.8 = 0.72
        # Rule contribution (bearishEngulfing) = 0.7 * 0.8 = 0.56
        # weighted_bearish_pattern_score = 0.72 + 0.56 = 1.28
        # is_strong_bearish_pattern = 1.28 >= 0.75 (strongPatternThreshold from mock) -> True
        # AI conf: GPT 0.6, Grok 0.65 -> combined_confidence = 0.625
        # exitPatternConfThreshold = 0.5 (from mock_params_get_for_tests)
        # Condition: is_strong_bearish_pattern (True) AND combined_confidence (0.625) > exitPatternConfThreshold (0.5) -> True. Exit.

        async def mock_ask_ai_moderate_conf_hold(*args, **kwargs):
            if 'gpt' in args[0].lower() if args else kwargs.get('context', {}).get('ai_type') == 'gpt': # crude way to differentiate if needed
                 return {"reflectie": "GPT: Hold, but watch out.", "confidence": 0.6, "intentie": "HOLD"}
            return {"reflectie": "Grok: Hold, but some concerns.", "confidence": 0.65, "intentie": "HOLD"}

        original_gpt_ask_temp = optimizer.gpt_reflector.ask_ai
        original_grok_ask_temp = optimizer.grok_reflector.ask_grok
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_moderate_conf_hold
        optimizer.grok_reflector.ask_grok = mock_ask_ai_moderate_conf_hold

        exit_decision_strong_pattern = await optimizer.should_exit(
            mock_df, mock_trade_profitable, test_symbol, test_strategy_id,
            learned_bias=0.5, learned_confidence=0.5, # Neutral bias/conf for direct param effect
            candles_by_timeframe=mock_candles_by_timeframe,
            additional_context_data=mock_additional_context_for_exit # Add this
        )
        print(f"Resultaat (Strong Bearish Pattern): {json.dumps(exit_decision_strong_pattern, indent=2, default=str)}")
        assert exit_decision_strong_pattern['exit'] is True
        assert exit_decision_strong_pattern['reason'] == "bearish_pattern_with_ai_confirmation"
        assert abs(exit_decision_strong_pattern['weighted_bearish_score'] - 1.28) < 0.01

        optimizer.gpt_reflector.ask_ai = original_gpt_ask_temp # Restore AI mocks
        optimizer.grok_reflector.ask_grok = original_grok_ask_temp

        # Scenario: Pattern Score Too Low - NO EXIT
        print("\n--- Test Scenario: Pattern Score Too Low - NO EXIT ---")
        # AI conf: GPT 0.6, Grok 0.65 -> combined_confidence = 0.625 (still > 0.5)
        # Intent: HOLD
        # CNN data (weak):
        # CNN score for f"{mock_df_timeframe}_no_bullFlag_score" = 0.1
        # Rule pattern "bearishEngulfing" = False
        # Calculation:
        # CNN contribution (no_bullFlag_score) = 0.1 * 0.8 = 0.08
        # Rule contribution = 0 (bearishEngulfing is False)
        # weighted_bearish_pattern_score = 0.08
        # is_strong_bearish_pattern = 0.08 >= 0.75 (strongPatternThreshold from mock) -> False
        # No exit trigger from pattern strength.
        # AI intent is HOLD, so no "ai_sell_intent_confident".
        # AI confidence 0.625 is not < exitConvictionDropTrigger (0.4), so no "low_ai_confidence_profit_taking".
        # Result: No exit.

        async def mock_ask_ai_hold_moderate_conf(*args, **kwargs): # Same as above, just for clarity
             if 'gpt' in args[0].lower() if args else kwargs.get('context', {}).get('ai_type') == 'gpt':
                 return {"reflectie": "GPT: Hold, market seems stable.", "confidence": 0.6, "intentie": "HOLD"}
             return {"reflectie": "Grok: Hold, nothing major.", "confidence": 0.65, "intentie": "HOLD"}

        weak_cnn_mock_data = {
            "patterns": {"bearishEngulfing": False, "eveningStar": False},
            "cnn_predictions": {
                f"{mock_df_timeframe}_no_bullFlag_score": 0.1, # Low bearish signal
                f"{mock_df_timeframe}_bullFlag_score": 0.9    # High bull signal (irrelevant for this path)
            }
        }
        original_cnn_detect_temp = optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = lambda *args, **kwargs: weak_cnn_mock_data.copy()

        original_gpt_ask_temp2 = optimizer.gpt_reflector.ask_ai
        original_grok_ask_temp2 = optimizer.grok_reflector.ask_grok
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_hold_moderate_conf
        optimizer.grok_reflector.ask_grok = mock_ask_ai_hold_moderate_conf

        exit_decision_low_score = await optimizer.should_exit(
            mock_df, mock_trade_profitable, test_symbol, test_strategy_id,
            learned_bias=0.5, learned_confidence=0.5,
            candles_by_timeframe=mock_candles_by_timeframe,
            additional_context_data=mock_additional_context_for_exit # Add this
        )
        print(f"Resultaat (Pattern Score Too Low): {json.dumps(exit_decision_low_score, indent=2, default=str)}")
        assert exit_decision_low_score['exit'] is False
        assert exit_decision_low_score['reason'] == "no_ai_exit_signal"
        assert abs(exit_decision_low_score['weighted_bearish_score'] - 0.08) < 0.01

        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = original_cnn_detect_temp # Restore CNN mock
        optimizer.gpt_reflector.ask_ai = original_gpt_ask_temp2 # Restore AI mocks
        optimizer.grok_reflector.ask_grok = original_grok_ask_temp2


        # Scenario: AI low confidence while in profit (ensure it still works with centralized mocks)
        print("\n--- Test Scenario: Low AI Confidence in Profit (Centralized Mocks) ---")
        # This scenario was previously "Scenario 2"
        # Params: exitConvictionDropTrigger = 0.4, minProfitForLowConfExit = 0.005
        # AI conf: GPT 0.3, Grok 0.3 -> combined_confidence = 0.3
        # Pattern score will be 1.28 from default_cnn_mock_data, but this path should trigger first.
        # Condition: combined_confidence (0.3) < exitConvictionDropTrigger (0.4) AND current_profit_pct (0.03) > minProfitForLowConfExit (0.005) -> True. Exit.

        async def mock_ask_ai_very_low_conf(*args, **kwargs):
            return {"reflectie": "Very uncertain.", "confidence": 0.3, "intentie": "HOLD"}

        original_gpt_ask_temp3 = optimizer.gpt_reflector.ask_ai
        original_grok_ask_temp3 = optimizer.grok_reflector.ask_grok
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_very_low_conf
        optimizer.grok_reflector.ask_grok = mock_ask_ai_very_low_conf
        # Ensure default CNN mock is active for this test (it should be unless overridden locally and not restored)
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = lambda *args, **kwargs: default_cnn_mock_data.copy()


        exit_decision_low_conf_profit = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, # These don't affect this path
            exit_conviction_drop_trigger=0.4, # Explicitly passed, but mock_params_get_for_tests also provides it
            candles_by_timeframe=mock_candles_by_timeframe,
            additional_context_data=mock_additional_context_for_exit # Add this
        )
        print(f"Resultaat (Low AI Conf Profit): {json.dumps(exit_decision_low_conf_profit, indent=2, default=str)}")
        assert exit_decision_low_conf_profit['exit'] is True
        assert exit_decision_low_conf_profit['reason'] == "low_ai_confidence_profit_taking"

        optimizer.gpt_reflector.ask_ai = original_gpt_ask_temp3 # Restore AI mocks
        optimizer.grok_reflector.ask_grok = original_grok_ask_temp3

        # Scenario: AI Sell Intent with Sufficient Confidence - EXIT
        print("\n--- Test Scenario: AI Sell Intent with Sufficient Confidence - EXIT ---")
        # Params: exitAISellIntentConfThreshold = 0.6
        # AI: GPT intent SELL (conf 0.7), Grok intent SELL (conf 0.65) -> ai_exit_intent = True, combined_confidence = 0.675
        # Pattern score is not the primary trigger here.
        # Condition: ai_exit_intent (True) AND combined_confidence (0.675) > exitAISellIntentConfThreshold (0.6) -> True. Exit.
        async def mock_ask_ai_strong_sell(*args, **kwargs):
            if 'gpt' in args[0].lower() if args else kwargs.get('context', {}).get('ai_type') == 'gpt':
                return {"reflectie": "GPT: Strong sell signal.", "confidence": 0.7, "intentie": "SELL"}
            return {"reflectie": "Grok: Looks bearish, sell.", "confidence": 0.65, "intentie": "SELL"}

        original_gpt_ask_temp4 = optimizer.gpt_reflector.ask_ai
        original_grok_ask_temp4 = optimizer.grok_reflector.ask_grok
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_strong_sell
        optimizer.grok_reflector.ask_grok = mock_ask_ai_strong_sell
        # Ensure default CNN mock is active
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = lambda *args, **kwargs: default_cnn_mock_data.copy()


        exit_decision_ai_sell = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.5, learned_confidence=0.5,
            candles_by_timeframe=mock_candles_by_timeframe,
            additional_context_data=mock_additional_context_for_exit # Add this
        )
        print(f"Resultaat (AI Sell Intent): {json.dumps(exit_decision_ai_sell, indent=2, default=str)}")
        assert exit_decision_ai_sell['exit'] is True
        assert exit_decision_ai_sell['reason'] == "ai_sell_intent_confident"

        optimizer.gpt_reflector.ask_ai = original_gpt_ask_temp4 # Restore AI mocks
        optimizer.grok_reflector.ask_grok = original_grok_ask_temp4

        # Restore general CNN mock if specific tests changed it and didn't restore
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = lambda *args, **kwargs: default_cnn_mock_data.copy()


        # --- Test optimize_trailing_stop_loss ---
        print("\n--- Test ExitOptimizer (optimize_trailing_stop_loss) ---")
        # The mock_params_get_for_tests provides defaults for SL params.
        optimizer.params_manager.get_param = mock_params_get_for_tests # Ensure general mock is active

        # Structure for SL test cases:
        # (test_name, gpt_reflection, gpt_confidence, grok_reflection, grok_confidence, expected_sl_pct_after_parsing_and_confidence_selection)
        # expected_sl_pct is the value like 0.025 for 2.5%
        sl_test_cases = [
            # Existing Regex Tests (should still pass, potentially via fallback)
            ("Regex_GPT_HighConf", "Adviseer stop loss op 2.5% voor deze trade.", 0.8, "No SL opinion.", 0.5, 0.025),
            ("Regex_Grok_HighConf", "No SL opinion.", 0.5, "Consider a stop-loss of 1.5 percent.", 0.7, 0.015),
            ("Regex_Both_GPTWins", "Trailing stop should be 2pct", 0.9, "SL at 5%", 0.8, 0.02),
            ("Regex_Both_GrokWins", "SL at 5%", 0.8, "Trailing stop should be 2.2pct", 0.9, 0.022),
            ("Regex_NoClearRecommendation", "No clear SL recommendation.", 0.6, "Definitely use a stop loss strategy.", 0.9, None),
            ("Regex_ValueOutOfRangeLow", "The sl is 0.05 %", 0.7, "No opinion", 0.5, None), # 0.0005, too low
            ("Regex_ValueOutOfRangeHigh", "Suggesting a stoploss of 60%.", 0.8, "No opinion", 0.5, None), # 0.60, too high

            # New JSON Tests
            ("JSON_GPT_Success", 'Explanation... {"recommended_sl_percentage": 3.5} ...more text', 0.85, "Regex SL: 2%", 0.7, 0.035),
            ("JSON_Grok_Success", "Regex SL: 5%", 0.7, 'Text then {"recommended_sl_percentage": 4.2} and more text', 0.82, 0.042),
            ("JSON_Both_GPTWins", 'GPT says {"recommended_sl_percentage": 2.8}', 0.9, 'Grok says {"recommended_sl_percentage": 3.0}', 0.88, 0.028),
            ("JSON_Both_GrokWins", 'GPT says {"recommended_sl_percentage": 2.8}', 0.88, 'Grok says {"recommended_sl_percentage": 3.0}', 0.9, 0.030),

            ("JSON_GPT_Malformed_Fallback_Grok_JSON", 'GPT has malformed {"recommended_sl_percentage": 2.5 ...', 0.8, 'Grok good: {"recommended_sl_percentage": 2.2}', 0.85, 0.022),
            ("JSON_GPT_Malformed_Fallback_Grok_Regex", 'GPT has malformed {"recommended_sl_percentage": 2.5 ...', 0.8, 'Grok regex SL 2.3%', 0.85, 0.023),

            ("JSON_GPT_Valid_Grok_Malformed_Fallback_GPT_JSON", 'GPT good: {"recommended_sl_percentage": 2.9}', 0.85, 'Grok malformed {"recommended_sl_percentage": 2.5 ...', 0.8, 0.029),
            ("JSON_GPT_Regex_Grok_Malformed_Fallback_GPT_Regex", 'GPT regex SL 2.8%', 0.85, 'Grok malformed {"recommended_sl_percentage": 2.5 ...', 0.8, 0.028),

            ("JSON_MissingKey_GPT_Fallback_Regex", 'GPT has {"some_other_key": 2.5}', 0.8, "Grok SL is 1.8%", 0.75, 0.018), # GPT JSON fails, Grok Regex wins
            ("JSON_MissingKey_Grok_Fallback_GPT_JSON", 'GPT good: {"recommended_sl_percentage": 3.3}', 0.85, 'Grok has {"another_key": 2.0}', 0.8, 0.033), # Grok JSON fails, GPT JSON wins

            ("JSON_ValueOutOfRange_GPT_Fallback_Grok_JSON", '{"recommended_sl_percentage": 0.01}', 0.8, 'Grok good: {"recommended_sl_percentage": 2.7}', 0.75, 0.027), # GPT JSON value too low
            ("JSON_ValueOutOfRange_Grok_Fallback_GPT_Regex", "GPT regex says SL 3.9%", 0.78, '{"recommended_sl_percentage": 75.0}', 0.8, 0.039), # Grok JSON value too high

            ("JSON_GPT_LowerConf_Grok_JSON_Wins", 'GPT: {"recommended_sl_percentage": 2.0}', 0.7, 'Grok: {"recommended_sl_percentage": 2.5}', 0.8, 0.025),
            ("JSON_GPT_LowerConf_Grok_Regex_Wins", 'GPT: {"recommended_sl_percentage": 2.0}', 0.7, 'Grok regex SL 2.5%', 0.8, 0.025),
            ("Regex_GPT_LowerConf_Grok_JSON_Wins", 'GPT regex SL 2.0%', 0.7, 'Grok: {"recommended_sl_percentage": 2.5}', 0.8, 0.025),
        ]

        original_gpt_ask_sl_temp = optimizer.gpt_reflector.ask_ai
        original_grok_ask_sl_temp = optimizer.grok_reflector.ask_grok

        for idx, (test_name, gpt_reflection, gpt_conf, grok_reflection, grok_conf, expected_sl_pct) in enumerate(sl_test_cases):
            print(f"\n--- SL Test Case #{idx + 1}: {test_name} ---")
            print(f"    GPT (conf {gpt_conf}): '{gpt_reflection}'")
            print(f"    Grok (conf {grok_conf}): '{grok_reflection}'")
            print(f"    Expected Parsed SL (0.0X for X%): {expected_sl_pct}")

            async def mock_gpt_sl_resp(*args, **kwargs):
                return {"reflectie": gpt_reflection, "confidence": gpt_conf, "intentie": "HOLD"}
            async def mock_grok_sl_resp(*args, **kwargs):
                return {"reflectie": grok_reflection, "confidence": grok_conf, "intentie": "HOLD"}

            optimizer.gpt_reflector.ask_ai = mock_gpt_sl_resp
            optimizer.grok_reflector.ask_grok = mock_grok_sl_resp

            # Ensure default SL params from mock_params_get_for_tests are used for clamping checks
            current_sl_min_offset = mock_params_get_for_tests("slMinOffset", default=0.005)
            current_sl_max_offset = mock_params_get_for_tests("slMaxOffset", default=0.10)

            sl_optimization_result = await optimizer.optimize_trailing_stop_loss(
                dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
                learned_bias=0.5, learned_confidence=0.5, candles_by_timeframe=mock_candles_by_timeframe,
                additional_context_data=mock_additional_context_for_exit # Add this
            )

            if expected_sl_pct is not None:
                assert sl_optimization_result is not None, f"Test Case '{test_name}' FAILED: Expected SL but got None."
                # final_trailing_offset is derived from ai_recommended_sl_pct.
                # With neutral bias/conf (0.5), factors are 1.0.
                # So, final_trailing_offset should be ai_recommended_sl_pct clamped by slMinOffset and slMaxOffset.
                clamped_expected_offset = max(current_sl_min_offset, min(expected_sl_pct, current_sl_max_offset))
                assert abs(sl_optimization_result['trailing_stop_positive_offset'] - clamped_expected_offset) < 0.0001, \
                    f"Test Case '{test_name}' FAILED: Expected offset ~{clamped_expected_offset:.4f} (using slMinOffset {current_sl_min_offset}), got {sl_optimization_result['trailing_stop_positive_offset']:.4f}."
                print(f"Test Case '{test_name}' PASSED. Parsed SL leading to offset: {sl_optimization_result['trailing_stop_positive_offset']:.4f} (Raw: {expected_sl_pct*100:.2f}%, Clamped by [{current_sl_min_offset*100:.2f}%, {current_sl_max_offset*100:.2f}%]: {clamped_expected_offset*100:.2f}%)")
            else:
                assert sl_optimization_result is None, f"Test Case '{test_name}' FAILED: Expected no SL (None) but got {sl_optimization_result}."
                print(f"Test Case '{test_name}' PASSED. No SL parsed as expected.")

        optimizer.gpt_reflector.ask_ai = original_gpt_ask_sl_temp
        optimizer.grok_reflector.ask_grok = original_grok_ask_sl_temp
        optimizer.params_manager.get_param = mock_params_get_for_tests # Restore general param mock

        # Restore original methods (master restoration)
        if optimizer.prompt_builder:
            optimizer.prompt_builder.generate_prompt_with_data = original_prompt_builder_generate


        # Scenario: Test custom SL adjustment parameters
        print(f"\n--- SL Test Case: Custom SL Adjustment Params ---")
        optimizer.params_manager.get_param = mock_params_get_for_tests # Ensure general mock is active for base params
        # Temporarily set specific params for this test using a lambda for get_param
        custom_params_for_sl_test = {
            "slBiasImpactFactor": 0.3, "slConfidenceImpactFactor": 0.3,
            "slMinOffset": 0.008, "slMaxOffset": 0.12,
            "slMinHardStop": -0.25, "slMaxHardStop": -0.02,
            "slDefaultTriggerProfit": 0.006,
            # Ensure other necessary params are available from the general mock
        }
        original_param_mock = optimizer.params_manager.get_param
        optimizer.params_manager.get_param = lambda key, strategy_id=None, default=None: custom_params_for_sl_test.get(key, mock_params_get_for_tests(key, strategy_id, default))

        # AI recommends 5% SL (JSON), neutral bias/confidence for this part of test
        async def mock_gpt_sl_json_5_pct(*args, **kwargs):
            return {"reflectie": 'Here is my advice: {"recommended_sl_percentage": 5.0}', "confidence": 0.8, "intentie": "HOLD"}
        async def mock_grok_no_opinion_sl(*args, **kwargs):
            return {"reflectie": "No SL opinion from Grok.", "confidence": 0.5, "intentie": "HOLD"}

        optimizer.gpt_reflector.ask_ai = mock_gpt_sl_json_5_pct
        optimizer.grok_reflector.ask_grok = mock_grok_no_opinion_sl

        sl_optimization_result_custom = await optimizer.optimize_trailing_stop_loss(
            mock_df, mock_trade_profitable, test_symbol, test_strategy_id,
            learned_bias=0.5, learned_confidence=0.5, # Neutral bias/conf
            candles_by_timeframe=mock_candles_by_timeframe,
            additional_context_data=mock_additional_context_for_exit # Add this
        )

        # Expected: ai_recommended_sl_pct = 0.05. Bias/Conf factors are 1.0.
        # final_trailing_offset = 0.05. Clamped by custom: max(0.008, min(0.05, 0.12)) = 0.05
        # hard_stoploss_value = -(0.05 * default hardStopMultiplier 1.5) = -0.075.
        # Clamped by custom hard stop: max(-0.25, min(-0.075, -0.02)) = -0.075
        # trailing_stop_trigger = custom slDefaultTriggerProfit = 0.006 (profit 0.03 < default slTriggerProfitThreshold 0.05)

        print(f"SL Optimalisatie Resultaat (Custom Params Test): {json.dumps(sl_optimization_result_custom, indent=2, default=str)}")
        assert sl_optimization_result_custom is not None
        assert abs(sl_optimization_result_custom['trailing_stop_positive_offset'] - 0.05) < 0.0001, f"Custom Param Test Offset: Expected 0.05, Got {sl_optimization_result_custom['trailing_stop_positive_offset']}"
        assert abs(sl_optimization_result_custom['stoploss'] - (-0.075)) < 0.0001, f"Custom Param Test Stoploss: Expected -0.075, Got {sl_optimization_result_custom['stoploss']}"
        assert abs(sl_optimization_result_custom['trailing_stop_positive'] - 0.006) < 0.0001, f"Custom Param Test Trigger: Expected 0.006, Got {sl_optimization_result_custom['trailing_stop_positive']}"
        print(f"SL Test Case Custom Params PASSED.")

        optimizer.params_manager.get_param = original_param_mock # Restore general param mock

        # Restore original methods (master restoration for all tests)
        if optimizer.prompt_builder: # Check as it might be None if init failed
            optimizer.prompt_builder.generate_prompt_with_data = original_prompt_builder_generate
        optimizer.gpt_reflector.ask_ai = original_gpt_ask
        optimizer.grok_reflector.ask_grok = original_grok_ask
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = original_cnn_detect
        optimizer.bias_reflector.get_bias_score = original_bias_get
        if optimizer.confidence_engine:
            optimizer.confidence_engine.get_confidence_score = original_conf_get
        optimizer.params_manager.get_param = original_params_get


    asyncio.run(run_test_exit_optimizer())
