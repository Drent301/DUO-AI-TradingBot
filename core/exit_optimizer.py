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
            logger.error(f"Failed to initialize PromptBuilder: {e}")
            self.prompt_builder = None
        self.bias_reflector = BiasReflector()
        try:
            self.confidence_engine = ConfidenceEngine()
        except Exception as e:
            logger.error(f"Failed to initialize ConfidenceEngine: {e}")
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
        candles_by_timeframe: Optional[Dict[str, pd.DataFrame]] = None # Relevant voor AI, made Optional with default
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
            )

        if not prompt:
            logger.warning(f"[ExitOptimizer] Geen prompt gegenereerd voor {symbol}. Exit evaluatie kan minder accuraat zijn.")
            prompt = f"Analyseer de huidige markt voor {symbol} en de openstaande trade ({trade.get('id','N/A')}) voor een mogelijk exit signaal. Huidige profit: {trade.get('profit_pct',0):.2%}"


        # Vraag AI-consensus
        gpt_response = await self.gpt_reflector.ask_ai(prompt, context={"symbol": symbol, "strategy_id": current_strategy_id, "trade": trade})
        grok_response = await self.grok_reflector.ask_grok(prompt, context={"symbol": symbol, "strategy_id": current_strategy_id, "trade": trade})

        gpt_confidence = gpt_response.get('confidence', 0.0) or 0.0
        grok_confidence = grok_response.get('confidence', 0.0) or 0.0
        gpt_intentie = str(gpt_response.get('intentie', '')).upper()
        grok_intentie = str(grok_response.get('intentie', '')).upper()

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

        # Fetch cnnPatternWeight from ParamsManager
        cnn_pattern_weight = self.params_manager.get_param("cnnPatternWeight", strategy_id=current_strategy_id, default=1.0)
        if not isinstance(cnn_pattern_weight, (float, int)): # Robust handling
            logger.warning(f"Invalid cnnPatternWeight type ({type(cnn_pattern_weight)}) for strategy {current_strategy_id}, using default 1.0.")
            cnn_pattern_weight = 1.0
        logger.info(f"Symbol: {symbol} (Exit), Using cnnPatternWeight: {cnn_pattern_weight} (Default: 1.0)")

        weighted_pattern_score = 0.0
        detected_patterns_summary = []
        cnn_predictions_data = pattern_data.get('cnn_predictions', {})
        rule_patterns_data = pattern_data.get('patterns', {})

        # 1. Process CNN Predictions for bearish signals
        cnn_prediction_timeframe = "15m" # As defined in cnn_patterns.py for predictions
        bearish_cnn_keys = [
            f"{cnn_prediction_timeframe}_no_bullFlag_score", # Proxy for bearish
            f"{cnn_prediction_timeframe}_bearSignal_score"   # Hypothetical specific bear signal
        ]

        if cnn_predictions_data: # Check if cnn_predictions dictionary exists and is not empty
            logger.info(f"Symbol: {symbol} (Exit), Evaluating CNN predictions: {cnn_predictions_data}")
            for key in bearish_cnn_keys:
                score = cnn_predictions_data.get(key)
                if score is not None and isinstance(score, (float, int)) and score > 0: # score > 0 indicates presence
                    contribution = score * cnn_pattern_weight
                    weighted_pattern_score += contribution
                    logger.info(f"Symbol: {symbol} (Exit), CNN Bearish contribution from '{key}': {score:.2f} * {cnn_pattern_weight:.2f} = {contribution:.2f}. Current weighted_score: {weighted_pattern_score:.2f}")
                    detected_patterns_summary.append(f"CNN_{key}({score:.2f})")
        else:
            logger.info(f"Symbol: {symbol} (Exit), No CNN predictions found in pattern_data.")

        # 2. Rule-Based Bearish Pattern Score
        exit_rule_pattern_score = self.params_manager.get_param("exitRulePatternScore", strategy_id=current_strategy_id, default=0.7)
        logger.info(f"Symbol: {symbol} (Exit), Using exitRulePatternScore: {exit_rule_pattern_score} (Default: 0.7)")

        rule_based_bearish_patterns = [
            'bearishEngulfing', 'eveningStar', 'threeBlackCrows', 'darkCloudCover',
            'bearishRSIDivergence', 'CDLEVENINGSTAR', 'CDL3BLACKCROWS', 'CDLDARKCLOUDCOVER',
            'CDLHANGINGMAN', 'doubleTop', 'descendingTriangle', 'parabolicCurveDown'
        ]

        if isinstance(rule_patterns_data, dict):
            for p_name in rule_based_bearish_patterns:
                if rule_patterns_data.get(p_name): # Checks for existence and truthiness
                    contribution = exit_rule_pattern_score * cnn_pattern_weight
                    weighted_pattern_score += contribution
                    logger.info(f"Symbol: {symbol} (Exit), Detected rule-based bearish pattern: '{p_name}'. Added contribution: {exit_rule_pattern_score:.2f} * {cnn_pattern_weight:.2f} = {contribution:.2f}. Current weighted_score: {weighted_pattern_score:.2f}")
                    detected_patterns_summary.append(f"rule({p_name})")
                    break # Only the first detected rule-based bearish pattern contributes
        else:
            logger.warning(f"Symbol: {symbol} (Exit), 'patterns' key missing or not a dict in pattern_data. Skipping rule-based pattern check.")

        if not any(s.startswith("rule(") for s in detected_patterns_summary) and not any(s.startswith("CNN_") for s in detected_patterns_summary):
             logger.info(f"Symbol: {symbol} (Exit), No contributing CNN or rule-based bearish patterns detected.")

        logger.info(f"Symbol: {symbol} (Exit), Final Calculated Weighted Bearish Pattern Score: {weighted_pattern_score:.2f} from patterns: [{'; '.join(detected_patterns_summary)}]")

        # Fetch strong_bearish_pattern_threshold from ParamsManager
        strong_bearish_pattern_threshold = self.params_manager.get_param(
            "strongBearishPatternThresholdExit",
            strategy_id=current_strategy_id,
            default=0.5 # Default value if not found in params.json
        )
        logger.info(f"Symbol: {symbol} (Exit), Using strongBearishPatternThresholdExit: {strong_bearish_pattern_threshold} (Strategy: {current_strategy_id}, Default: 0.5)")

        is_strong_bearish_pattern = weighted_pattern_score >= strong_bearish_pattern_threshold

        log_msg_pattern_strength = f"Strong bearish pattern {'DETECTED' if is_strong_bearish_pattern else 'NOT detected'}. Score {weighted_pattern_score:.2f} {' >=' if is_strong_bearish_pattern else ' <'} Threshold {strong_bearish_pattern_threshold:.2f}"
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
        candles_by_timeframe: Optional[Dict[str, pd.DataFrame]] = None # Made Optional with default
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
            )
        if not prompt:
            logger.warning(f"[ExitOptimizer] Geen risicomanagement prompt gegenereerd voor {symbol}. SL niet geoptimaliseerd.")
            return None

        # Vraag AI om SL-advies
        gpt_response = await self.gpt_reflector.ask_ai(prompt, context={"symbol": symbol, "strategy_id": current_strategy_id, "trade": trade})
        grok_response = await self.grok_reflector.ask_grok(prompt, context={"symbol": symbol, "strategy_id": current_strategy_id, "trade": trade})

        ai_recommended_sl_pct: Optional[float] = None
        highest_confidence_for_sl = 0.0
        sl_keyword_mentioned = False

        # Define a list of regex patterns, from more specific to more general
        # Keywords: stop loss, stoploss, sl, s.l, trailing stop, tsl, trailing_stop
        # Units: %, pct, percent
        # Structure: keyword [connector] value [unit]
        sl_regex_patterns = [
            # Specific: "stop loss at 2.5%", "SL:3%", "trailing_stop_of_1.5pct"
            re.compile(r"(?:stop(?:[-_\s]?loss)?|s(?:[-_\s]?l)?|trailing(?:[-_\s]?stop)?|tsl)\s*(?:at|is|to|of|should be|[:])?\s*(\d+(?:\.\d+)?)\s*(?:%|pct|percent)\b", re.IGNORECASE),
            # General: "5 % stop loss", "value percent trailing stop" - value first
            re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|pct|percent)\s*(?:stop(?:[-_\s]?loss)?|s(?:[-_\s]?l)?|trailing(?:[-_\s]?stop)?|tsl)\b", re.IGNORECASE),
            # Broader, just number near keyword, hoping unit is implied or contextually clear (less reliable)
            re.compile(r"(?:stop(?:[-_\s]?loss)?|s(?:[-_\s]?l)?|trailing(?:[-_\s]?stop)?|tsl)\s*.*?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE),
        ]
        sl_keyword_regex = re.compile(r"\b(?:stop(?:[-_\s]?loss)?|s(?:[-_\s]?l)?|trailing(?:[-_\s]?stop)?|tsl)\b", re.IGNORECASE)


        for i, resp in enumerate([gpt_response, grok_response]):
            ai_name = "GPT" if i == 0 else "Grok"
            reflection = resp.get('reflectie', '')
            resp_confidence = resp.get('confidence', 0.0) or 0.0

            if sl_keyword_regex.search(reflection):
                sl_keyword_mentioned = True

            if resp_confidence > 0.55: # Threshold for considering SL advice
                parsed_sl_value_for_this_response = None
                for pattern_idx, sl_regex in enumerate(sl_regex_patterns):
                    sl_match = sl_regex.search(reflection)
                    if sl_match:
                        try:
                            sl_value_str = sl_match.group(1)
                            sl_value_float = float(sl_value_str) / 100.0 # Convert to percentage

                            # Sanity checks for the parsed SL percentage
                            if 0.001 <= sl_value_float <= 0.50: # e.g., 0.1% to 50%
                                logger.info(f"[{ai_name}] Parsed SL value: {sl_value_float:.2%} from reflection: '{reflection}' using regex pattern #{pattern_idx+1}. Confidence: {resp_confidence:.2f}")
                                # Prioritize based on confidence and if it's a better (more confident) recommendation
                                if resp_confidence > highest_confidence_for_sl:
                                    parsed_sl_value_for_this_response = sl_value_float
                                    highest_confidence_for_sl = resp_confidence
                                break # Found a match with this regex, no need to try others for this response
                            else:
                                logger.warning(f"[{ai_name}] Parsed SL value {sl_value_float:.2%} is outside reasonable range (0.1%-50%). Ignored. Reflection: '{reflection}'")
                        except ValueError:
                            logger.warning(f"[{ai_name}] Kon SL percentage niet parsen (ValueError) uit '{sl_match.group(1)}' in reflectie: '{reflection}'")
                        except Exception as e:
                            logger.error(f"[{ai_name}] Onverwachte fout bij parsen SL uit '{sl_match.group(1)}': {e}. Reflectie: '{reflection}'")

                if parsed_sl_value_for_this_response is not None:
                    # If this response yields a valid SL and has higher confidence, it becomes the current best
                     ai_recommended_sl_pct = parsed_sl_value_for_this_response
            else:
                logger.debug(f"[{ai_name}] AI response confidence {resp_confidence:.2f} not high enough for SL parsing, or no SL value found.")

        if ai_recommended_sl_pct is None:
            if sl_keyword_mentioned:
                logger.warning(f"[ExitOptimizer] AI mentioned stop-loss keywords but a specific percentage could not be parsed reliably from responses for {symbol}.")
            else:
                logger.info(f"[ExitOptimizer] Geen AI-advies voor SL-optimalisatie (geen SL percentage geparsed) voor {symbol}.")
            return None

        # If ai_recommended_sl_pct has a value, proceed with logic
        logger.info(f"[ExitOptimizer] Hoogst vertrouwd AI-aanbevolen SL: {ai_recommended_sl_pct:.2%} (confidence: {highest_confidence_for_sl:.2f}) voor {symbol}.")

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

        # Mock params_manager.get_param to return specific values for keys
        current_mock_params = {} # Make this a dictionary accessible by the test cases
        def mock_get_param_dynamic(key, strategy_id=None, default=None):
            # Allow tests to set params via current_mock_params dictionary
            return current_mock_params.get(key, default)
        optimizer.params_manager.get_param = mock_get_param_dynamic


        # --- Test should_exit ---
        print("\n--- Test ExitOptimizer (should_exit) ---")

        # Scenario 1A: Strong Bearish Pattern (Rule + CNN), Custom Threshold (0.6) - Exit
        print("\n--- Test Scenario 1A: Strong Bearish Pattern (Rule + CNN), Custom Threshold (0.6) - EXIT ---")
        current_mock_params = {
            "exitConvictionDropTrigger": 0.4,
            "cnnPatternWeight": 0.8, # Custom weight
            "strongBearishPatternThresholdExit": 0.6,
            "exitPatternConfThreshold": 0.5,
            "exitAISellIntentConfThreshold": 0.6,
            "minProfitForLowConfExit": 0.005,
            "exitRulePatternScore": 0.7 # Default rule score
        }
        optimizer.params_manager.get_param = lambda key, strategy_id=None, default=None: current_mock_params.get(key, default)


        async def mock_ask_ai_s1a(*args, **kwargs): # High AI confidence
            return {"reflectie": "Hold, but watch out.", "confidence": 0.7, "intentie": "HOLD"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_s1a
        optimizer.grok_reflector.ask_grok = mock_ask_ai_s1a

        async def mock_cnn_s1a(*args, **kwargs):
            return {
                "patterns": {"eveningStar": True}, # Rule pattern
                "cnn_predictions": {"15m_no_bullFlag_score": 0.9} # CNN bearish score
            }
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_s1a

        # Calculation for S1A:
        # CNN_contrib = 0.9 (score) * 0.8 (weight) = 0.72
        # Rule_contrib = 0.7 (exitRulePatternScore) * 0.8 (weight) = 0.56 (eveningStar is first)
        # weighted_pattern_score = 0.72 + 0.56 = 1.28
        # strong_bearish_pattern_threshold = 0.6 (from current_mock_params)
        # is_strong_bearish_pattern = 1.28 >= 0.6 -> True
        # AI_conf (0.7) > exitPatternConfThreshold (0.5) -> True. Exit.
        exit_decision_s1a = await optimizer.should_exit(
            mock_df, mock_trade_profitable, test_symbol, test_strategy_id, 0.5, 0.7, 0.4, mock_candles_by_timeframe)
        print(f"Resultaat (Scenario 1A): {json.dumps(exit_decision_s1a, indent=2, default=str)}")
        assert exit_decision_s1a['exit'] is True
        assert exit_decision_s1a['reason'] == "bearish_pattern_with_ai_confirmation"
        assert abs(exit_decision_s1a['weighted_bearish_score'] - 1.28) < 0.01

        # Scenario 1B: Weak Pattern (Low CNN, No Rule), Custom Threshold (0.8) - NO Exit
        print("\n--- Test Scenario 1B: Weak Pattern (Low CNN, No Rule), Custom Threshold (0.8) - NO EXIT ---")
        current_mock_params = {
            "exitConvictionDropTrigger": 0.4,
            "cnnPatternWeight": 1.0,
            "strongBearishPatternThresholdExit": 0.8, # Higher threshold
            "exitPatternConfThreshold": 0.5,
            "exitAISellIntentConfThreshold": 0.6,
            "minProfitForLowConfExit": 0.005,
            "exitRulePatternScore": 0.7
        }
        optimizer.params_manager.get_param = lambda key, strategy_id=None, default=None: current_mock_params.get(key, default)

        async def mock_cnn_s1b(*args, **kwargs):
            return {
                "patterns": {}, # No rule pattern
                "cnn_predictions": {"15m_no_bullFlag_score": 0.2} # Low CNN score
            }
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_s1b
        # Using AI mock from S1A (AI conf 0.7 > 0.5)
        # Calculation for S1B:
        # CNN_contrib = 0.2 * 1.0 = 0.2
        # Rule_contrib = 0
        # weighted_pattern_score = 0.2
        # strong_bearish_pattern_threshold = 0.8
        # is_strong_bearish_pattern = 0.2 >= 0.8 -> False. No Exit from this rule.
        exit_decision_s1b = await optimizer.should_exit(
            mock_df, mock_trade_profitable, test_symbol, test_strategy_id, 0.5, 0.7, 0.4, mock_candles_by_timeframe)
        print(f"Resultaat (Scenario 1B): {json.dumps(exit_decision_s1b, indent=2, default=str)}")
        assert exit_decision_s1b['exit'] is False

        # Scenario 1C: Strong Rule Pattern, Default Thresholds (exitRulePatternScore=0.7, strongBearishPatternThresholdExit=0.5), No CNN - Exit
        print("\n--- Test Scenario 1C: Strong Rule Pattern, Default Thresholds, No CNN - EXIT ---")
        current_mock_params = { # Defaults will be used for unspecified params here
            "exitConvictionDropTrigger": 0.4,
            "cnnPatternWeight": 1.0, # Default cnnPatternWeight
            # strongBearishPatternThresholdExit will use default 0.5
            # exitRulePatternScore will use default 0.7
        }
        optimizer.params_manager.get_param = lambda key, strategy_id=None, default=None: current_mock_params.get(key, default)

        async def mock_cnn_s1c(*args, **kwargs):
            return {"patterns": {"CDL3BLACKCROWS": True}, "cnn_predictions": {}} # No CNN score
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_s1c
        # Using AI mock from S1A (AI conf 0.7 > exitPatternConfThreshold default 0.5)
        # Calculation for S1C:
        # CNN_contrib = 0
        # Rule_contrib = 0.7 (default exitRulePatternScore) * 1.0 (default cnnPatternWeight) = 0.7
        # weighted_pattern_score = 0.7
        # strong_bearish_pattern_threshold = 0.5 (default)
        # is_strong_bearish_pattern = 0.7 >= 0.5 -> True. Exit.
        exit_decision_s1c = await optimizer.should_exit(
            mock_df, mock_trade_profitable, test_symbol, test_strategy_id, 0.5, 0.7, 0.4, mock_candles_by_timeframe)
        print(f"Resultaat (Scenario 1C): {json.dumps(exit_decision_s1c, indent=2, default=str)}")
        assert exit_decision_s1c['exit'] is True
        assert exit_decision_s1c['reason'] == "bearish_pattern_with_ai_confirmation"
        assert abs(exit_decision_s1c['weighted_bearish_score'] - 0.7) < 0.01


        # Scenario 2: AI low confidence while in profit (original test, ensure default threshold doesn't break it)
        print("\n--- Test Scenario 2: Low AI Confidence in Profit (Default Threshold context) ---")
        async def mock_ask_ai_low_conf(*args, **kwargs):
            return {"reflectie": "Onzeker beeld.", "confidence": 0.3, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_low_conf
        optimizer.grok_reflector.ask_grok = mock_ask_ai_low_conf
        async def mock_cnn_no_patterns(*args, **kwargs): return {"patterns": {}, "cnn_predictions": {}} # No patterns
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_no_patterns

        # weighted_pattern_score will be 0.0. is_strong_bearish_pattern = False.
        # combined_confidence = 0.3. exit_conviction_drop_trigger = 0.4. current_profit_pct = 0.03 (profitable)
        # 0.3 < 0.4 is True. Exit.
        exit_decision_s2 = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print(f"Resultaat (Scenario 2 - Low AI Conf Profit): {json.dumps(exit_decision_s2, indent=2, default=str)}")
        assert exit_decision_s2['exit'] is True
        assert exit_decision_s2['reason'] == "low_ai_confidence_profit_taking"

        # Scenario 3: No strong signal to exit (weighted score too low, only weak CNN)
        print("\n--- Test Scenario 3: No Strong Signal (Weak CNN, No Rule) ---")
        async def mock_ask_ai_hold(*args, **kwargs):
            return {"reflectie": "Markt stabiel.", "confidence": 0.7, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_hold
        optimizer.grok_reflector.ask_grok = mock_ask_ai_hold
        async def mock_cnn_weak_bearish_cnn_only(*args, **kwargs):
            return {"patterns": {}, "cnn_predictions": {"15m_no_bullFlag_score": 0.2}} # Weak CNN signal
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_weak_bearish_cnn_only

        # Params: cnnPatternWeight = 1.0.
        # weighted_pattern_score = (0.2 from CNN) * 1.0 = 0.2.
        # is_strong_bearish_pattern = False (0.2 < 0.5).
        # AI confidence 0.7, intent HOLD. Not low_ai_confidence_profit_taking as profit is negative.
        # No exit.
        exit_decision_s3 = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_losing, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print(f"Resultaat (Scenario 3 - No Strong Signal, Weak CNN): {json.dumps(exit_decision_s3, indent=2, default=str)}")
        assert exit_decision_s3['exit'] is False
        assert exit_decision_s3['reason'] == "no_ai_exit_signal"
        assert abs(exit_decision_s3['weighted_bearish_score'] - 0.2) < 0.01


        # --- Test optimize_trailing_stop_loss ---
        print("\n--- Test ExitOptimizer (optimize_trailing_stop_loss) ---")

        sl_test_cases = [
            ("Adviseer stop loss op 2.5% voor deze trade.", 0.8, 0.025),
            ("Consider a stop-loss of 1.5 percent.", 0.7, 0.015),
            ("Trailing stop should be 2pct", 0.9, 0.02),
            ("Market is volatile, maybe a stoploss around 5 % is good.", 0.6, 0.05),
            ("SL:3.2%", 0.85, 0.032),
            ("Set S.L to 4.0 percent", 0.75, 0.04),
            ("trailing_stop_of_1.2pct", 0.82, 0.012),
            ("My analysis suggests a stop loss of 0.8 %.", 0.9, 0.008), # Test value < 1% but still valid
            ("A 10 percent stop-loss seems appropriate here.", 0.8, 0.10),
            ("No clear SL recommendation.", 0.5, None), # Should not parse
            ("Definitely use a stop loss strategy.", 0.9, None), # Should not parse a value
            ("The sl is 0.05 %", 0.7, None), # Should parse 0.0005, which is outside 0.1%-50% range
            ("Suggesting a stoploss of 60%.", 0.8, None), # Outside 0.1%-50% range
            ("TSL to 7.5 percent.", 0.88, 0.075),
            ("stoploss:15pct", 0.92, 0.15),
            ("Stop loss of around three point five percent", 0.6, None), # Word numbers not supported by current regex
        ]

        for idx, (reflection_text, confidence, expected_sl_pct) in enumerate(sl_test_cases):
            print(f"\n--- SL Test Case #{idx + 1}: Reflection: '{reflection_text}', Confidence: {confidence} ---")

            async def mock_ask_ai_variable_sl(*args, **kwargs):
                return {"reflectie": reflection_text, "confidence": confidence, "intentie": "HOLD", "emotie": "neutraal"}

            # Apply this mock to one of the AIs, the other can be a non-parsing one or same
            optimizer.gpt_reflector.ask_ai = mock_ask_ai_variable_sl
            async def mock_ask_grok_no_sl(*args, **kwargs): # Ensure Grok doesn't interfere unless intended
                return {"reflectie": "No specific SL value from Grok.", "confidence": 0.5, "intentie": "HOLD"}
            optimizer.grok_reflector.ask_grok = mock_ask_grok_no_sl

            sl_optimization_result = await optimizer.optimize_trailing_stop_loss(
                dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
                learned_bias=0.5, learned_confidence=0.5, candles_by_timeframe=mock_candles_by_timeframe # Neutral bias/conf for easier testing of parsed SL
            )

            if expected_sl_pct is not None:
                assert sl_optimization_result is not None, f"Test Case #{idx+1} FAILED: Expected SL but got None. Reflection: '{reflection_text}'"
                assert 'stoploss' in sl_optimization_result
                assert 'trailing_stop_positive_offset' in sl_optimization_result
                # The final_trailing_offset is derived from ai_recommended_sl_pct after bias/confidence factors.
                # With neutral bias/conf (0.5), bias_factor and confidence_factor are 1.0.
                # So, final_trailing_offset should be clamped ai_recommended_sl_pct.
                # Clamping is max(0.005, min(expected_sl_pct, 0.10))
                clamped_expected_offset = max(0.005, min(expected_sl_pct, 0.10))
                assert abs(sl_optimization_result['trailing_stop_positive_offset'] - clamped_expected_offset) < 0.0001, \
                    f"Test Case #{idx+1} FAILED: Expected offset ~{clamped_expected_offset:.4f}, got {sl_optimization_result['trailing_stop_positive_offset']:.4f}. Reflection: '{reflection_text}'"
                print(f"Test Case #{idx+1} PASSED. Parsed SL leading to offset: {sl_optimization_result['trailing_stop_positive_offset']:.4f} (Expected raw: {expected_sl_pct}, Clamped: {clamped_expected_offset:.4f})")
            else:
                assert sl_optimization_result is None, f"Test Case #{idx+1} FAILED: Expected no SL (None) but got {sl_optimization_result}. Reflection: '{reflection_text}'"
                print(f"Test Case #{idx+1} PASSED. No SL parsed as expected. Reflection: '{reflection_text}'")

        # Restore original methods before this specific test case
        optimizer.params_manager.get_param = mock_get_param_dynamic # Ensure dynamic mock is active

        # Scenario: Test custom SL adjustment parameters
        print(f"\n--- SL Test Case: Custom SL Adjustment Params ---")
        current_mock_params = {
            "slBiasImpactFactor": 0.3, # Default 0.2
            "slConfidenceImpactFactor": 0.3, # Default 0.2
            "slMinOffset": 0.008, # Default 0.005
            "slMaxOffset": 0.12,  # Default 0.10
            "slMinHardStop": -0.25, # Default -0.20
            "slMaxHardStop": -0.02, # Default -0.01
            "slDefaultTriggerProfit": 0.006 # Default 0.005
        }
        # AI recommends 5% SL, neutral bias/confidence for this part of test
        async def mock_ask_ai_sl_5_pct(*args, **kwargs):
            return {"reflectie": "Suggest SL of 5%", "confidence": 0.8, "intentie": "HOLD"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_sl_5_pct
        optimizer.grok_reflector.ask_grok = mock_ask_grok_no_sl # Grok doesn't interfere

        sl_optimization_result_custom = await optimizer.optimize_trailing_stop_loss(
            mock_df, mock_trade_profitable, test_symbol, test_strategy_id,
            learned_bias=0.5, learned_confidence=0.5, # Neutral bias/conf
            candles_by_timeframe=mock_candles_by_timeframe
        )
        # Expected: ai_recommended_sl_pct = 0.05. Bias/Conf factors are 1.0.
        # final_trailing_offset = 0.05. Clamped by custom: max(0.008, min(0.05, 0.12)) = 0.05
        # hard_stoploss_value = -(0.05 * 1.5) = -0.075. Clamped by custom: max(-0.25, min(-0.075, -0.02)) = -0.075
        # trailing_stop_trigger = custom slDefaultTriggerProfit = 0.006 (profit 0.03 < 0.05)
        print(f"SL Optimalisatie Resultaat (Custom Params): {json.dumps(sl_optimization_result_custom, indent=2, default=str)}")
        assert sl_optimization_result_custom is not None
        assert abs(sl_optimization_result_custom['trailing_stop_positive_offset'] - 0.05) < 0.0001
        assert abs(sl_optimization_result_custom['stoploss'] - (-0.075)) < 0.0001
        assert abs(sl_optimization_result_custom['trailing_stop_positive'] - 0.006) < 0.0001
        print(f"SL Test Case Custom Params PASSED.")


        # Restore original methods
        if optimizer.prompt_builder:
            optimizer.prompt_builder.generate_prompt_with_data = original_prompt_builder_generate
        optimizer.gpt_reflector.ask_ai = original_gpt_ask
        optimizer.grok_reflector.ask_grok = original_grok_ask
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = original_cnn_detect
        optimizer.bias_reflector.get_bias_score = original_bias_get
        if optimizer.confidence_engine:
            optimizer.confidence_engine.get_confidence_score = original_conf_get
        optimizer.params_manager.get_param = original_params_get


    asyncio.run(run_test_exit_optimizer())
