# core/ai_activation_engine.py
import logging
import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
# import numpy as np # No longer needed for mock data generation here
# import dotenv # No longer needed for __main__

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
# logger.setLevel(logging.INFO) # Logging level configured by application

from pathlib import Path # Import Path

# Padconfiguratie met pathlib
CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CORE_DIR.parent # Assumes 'core' is directly under project root
MEMORY_DIR = (PROJECT_ROOT_DIR / 'user_data' / 'memory').resolve() # Ensure absolute path
REFLECTIE_LOG_FILE = (MEMORY_DIR / 'reflectie-logboek.json').resolve() # Same log file as ReflectieLus

MEMORY_DIR.mkdir(parents=True, exist_ok=True)

class AIActivationEngine:
    """
    Trigger-engine voor AI bij events: entry, exit, reflection, CNN-triggers.
    Vertaald en geoptimaliseerd van aiActivationEngine.js.
    """

    def __init__(self, reflectie_lus_instance: ReflectieLus):
        try:
            self.prompt_builder = PromptBuilder()
        except Exception as e:
            logger.error(f"Failed to initialize PromptBuilder in AIActivationEngine: {e}", exc_info=True)
            self.prompt_builder = None # Fallback

        self.reflectie_lus_instance = reflectie_lus_instance
        if self.reflectie_lus_instance is None:
            logger.error("ReflectieLus instance not provided to AIActivationEngine. AI activation will not work correctly.", exc_info=True) # Consider if exc_info is needed if not an exception
            # Optionally raise an error: raise ValueError("ReflectieLus instance is required")

        self.gpt_reflector = GPTReflector()
        self.grok_reflector = GrokReflector()
        self.cnn_patterns_detector = CNNPatterns()

        try:
            self.grok_sentiment_fetcher = GrokSentimentFetcher()
        except ValueError as e:
            logger.error(f"Failed to initialize GrokSentimentFetcher: {e}. Sentiment analysis will be disabled.", exc_info=True)
            self.grok_sentiment_fetcher = None
        except Exception as e_general: # Catch other potential errors during init
            logger.error(f"Unexpected error initializing GrokSentimentFetcher: {e_general}. Sentiment analysis will be disabled.", exc_info=True)
            self.grok_sentiment_fetcher = None


        self.last_reflection_timestamps: Dict[str, float] = {} # Per token
        # CNN_DETECTION_THRESHOLD is used as self.CNN_DETECTION_THRESHOLD
        # It should be defined as a class attribute.
        logger.info("AIActivationEngine geïnitialiseerd.")

    CNN_DETECTION_THRESHOLD = 0.7 # Class constant

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
                    trigger_data['fetched_sentiment_items'] = sentiment_items # Store fetched items
                else:
                    logger.info(f"No sentiment items found for {symbol}.")
            except Exception as e:
                logger.error(f"Error fetching sentiment for {symbol}: {e}", exc_info=True)
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
            if REFLECTIE_LOG_FILE.exists() and REFLECTIE_LOG_FILE.stat().st_size > 0:
                with REFLECTIE_LOG_FILE.open('r', encoding='utf-8') as f:
                    logs = json.load(f)
                if not isinstance(logs, list):
                    logger.warning(f"Reflectie logboek {REFLECTIE_LOG_FILE} bevatte geen lijst. Wordt opnieuw gestart als lijst.")
                    logs = []
        except json.JSONDecodeError:
            logger.warning(f"Reflectie logboek {REFLECTIE_LOG_FILE} is leeg of corrupt. Start met lege lijst.", exc_info=True)
            logs = [] # Ensure logs is a list
        except FileNotFoundError: # Should be caught by .exists() but good practice
             logger.warning(f"Reflectie logboek {REFLECTIE_LOG_FILE} niet gevonden. Start met lege lijst.", exc_info=True)
             logs = [] # Ensure logs is a list
        except Exception as e_generic: # Catch any other reading error
            logger.error(f"Onverwachte fout bij lezen {REFLECTIE_LOG_FILE}: {e_generic}", exc_info=True)
            logs = [] # Ensure logs is a list


        logs.append(entry)
        try:
            def save_log_sync():
                with REFLECTIE_LOG_FILE.open('w', encoding='utf-8') as f:
                    json.dump(logs, f, indent=2)
            await asyncio.to_thread(save_log_sync)
        except IOError as e:
            logger.error(f"Fout bij opslaan reflectielogboek naar {REFLECTIE_LOG_FILE}: {e}", exc_info=True)
        except Exception as e_generic_save: # Catch any other saving error
            logger.error(f"Onverwachte fout bij opslaan reflectielogboek naar {REFLECTIE_LOG_FILE}: {e_generic_save}", exc_info=True)


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
            logger.error("PromptBuilder niet geïnitialiseerd in AIActivationEngine. Kan AI niet activeren.", exc_info=False) # No exception context here
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
        # We need to add trigger_type, pattern_data, and potentially sentiment_data.
        extended_trade_context = {
            **(trade_context or {}),
            "trigger_type": trigger_type, # Pass the original trigger_type
        }

        sentiment_data_for_prompt = trigger_data.get('fetched_sentiment_items')
        if sentiment_data_for_prompt:
            extended_trade_context['social_sentiment_data'] = sentiment_data_for_prompt
            logger.info(f"Added {len(sentiment_data_for_prompt)} sentiment items to extended_trade_context for {token}.")

        logger.info(f"[AI-ACTIVATION] Delegating to ReflectieLus for {token} ({trigger_type}). Context keys: {list(extended_trade_context.keys())}")

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
            prompt_type=prompt_type, # Use the determined prompt_type
            pattern_data=pattern_data,
            bias_reflector_instance=bias_reflector_instance, # Pass through
            confidence_engine_instance=confidence_engine_instance # Pass through
        )

        # The original activate_ai returned reflectie_log_entry.
        # process_reflection_cycle returns a dictionary which might be different.
        # We return what ReflectieLus returns.
        return reflection_result
