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
        cnn_pattern_weight = self.params_manager.get_param("cnnPatternWeight", strategy_id=current_strategy_id)
        if cnn_pattern_weight is None:
            cnn_pattern_weight = 1.0 # Default value if not found
            logger.warning(f"cnnPatternWeight not found for strategy {current_strategy_id} in ExitOptimizer, using default 1.0.")

        weighted_pattern_score = 0.0
        detected_patterns_summary = [] # For logging

        # 1. Numerical CNN Bearish Scores - Placeholder as cnn_patterns.py doesn't provide specific bearish ML scores yet.
        #    If pattern_data['patterns'] had 'ml_cnn_bearish_score':
        #    ml_bearish_score = pattern_data.get('patterns', {}).get('ml_cnn_bearish_score', 0.0)
        #    if isinstance(ml_bearish_score, (float, int)) and ml_bearish_score > 0:
        #        contribution = ml_bearish_score * cnn_pattern_weight
        #        weighted_pattern_score += contribution
        #        detected_patterns_summary.append(f"ml_bearish({ml_bearish_score:.2f})")

        # 2. Rule-Based Bearish Pattern Score
        rule_based_bearish_patterns = [
            'bearishEngulfing', 'eveningStar', 'threeBlackCrows', 'darkCloudCover',
            'bearishRSIDivergence', 'CDLEVENINGSTAR', 'CDL3BLACKCROWS', 'CDLDARKCLOUDCOVER',
            'CDLHANGINGMAN', 'doubleTop', 'descendingTriangle', 'parabolicCurveDown'
        ]

        rule_pattern_detected_flag = False
        detected_rule_patterns_log = []
        if pattern_data and isinstance(pattern_data.get('patterns'), dict):
            for p_name in rule_based_bearish_patterns:
                if pattern_data['patterns'].get(p_name): # Checks for existence and truthiness
                    rule_pattern_detected_flag = True
                    detected_rule_patterns_log.append(p_name)
                    break  # Add contribution only once

        if rule_pattern_detected_flag:
            rule_based_contribution_value = 0.7
            contribution = rule_based_contribution_value * cnn_pattern_weight
            weighted_pattern_score += contribution
            logger.info(f"Symbol: {symbol} (Exit), Detected rule-based bearish patterns: {detected_rule_patterns_log}. Added contribution: {rule_based_contribution_value:.2f} * {cnn_pattern_weight:.2f} = {contribution:.2f}.")
            detected_patterns_summary.append(f"rules({','.join(detected_rule_patterns_log)})")
        else:
            logger.info(f"Symbol: {symbol} (Exit), No strong rule-based bearish patterns detected from the predefined list.")

        logger.info(f"Symbol: {symbol} (Exit), Final Calculated Weighted Bearish Pattern Score: {weighted_pattern_score:.2f} from patterns: [{'; '.join(detected_patterns_summary)}]")

        strong_bearish_pattern_threshold = 0.5 # Consider making this configurable
        is_strong_bearish_pattern = weighted_pattern_score >= strong_bearish_pattern_threshold

        log_msg_pattern_strength = f"Strong bearish pattern {'DETECTED' if is_strong_bearish_pattern else 'NOT detected'}. Score {weighted_pattern_score:.2f} {' >=' if is_strong_bearish_pattern else ' <'} Threshold {strong_bearish_pattern_threshold:.2f}"
        logger.info(f"Symbol: {symbol} (Exit), {log_msg_pattern_strength}")

        # AI-besluitvormingslogica voor exit
        current_profit_pct = trade.get('profit_pct', 0.0)
        # exit_conviction_drop_trigger is passed as argument, or use default from signature

        # Scenario 1: AI is onzeker en trade is in winst (take profit)
        if combined_confidence < exit_conviction_drop_trigger and current_profit_pct > 0.005:
            logger.info(f"[ExitOptimizer] ✅ Exit door lage AI confidence ({combined_confidence:.2f} < {exit_conviction_drop_trigger}) en trade in winst ({current_profit_pct:.2%}) voor {symbol}.")
            return {"exit": True, "reason": "low_ai_confidence_profit_taking", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {}), "weighted_bearish_score": weighted_pattern_score}

        # Scenario 2: Sterk bearish patroon met AI-bevestiging
        if is_strong_bearish_pattern and combined_confidence > 0.5: # AI confirms with at least moderate confidence
             logger.info(f"[ExitOptimizer] ✅ Exit door {log_msg_pattern_strength} en AI confidence {combined_confidence:.2f} voor {symbol}.")
             return {"exit": True, "reason": "bearish_pattern_with_ai_confirmation", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {}), "weighted_bearish_score": weighted_pattern_score}

        # Scenario 3: AI wil verkopen met voldoende confidence
        if ai_exit_intent and combined_confidence > 0.6:
            logger.info(f"[ExitOptimizer] ✅ Exit door AI verkoop intentie (GPT: {gpt_intentie}, Grok: {grok_intentie}) met confidence {combined_confidence:.2f} voor {symbol}.")
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

        for resp in [gpt_response, grok_response]:
            resp_confidence = resp.get('confidence', 0.0) or 0.0
            if resp_confidence > 0.55 and resp_confidence > highest_confidence_for_sl: # Threshold for considering SL advice
                sl_match = re.search(r"(?:stop loss|stoploss|sl|trailing stop|tsl)\s*(?:at|is|to|should be|of)?\s*(\d+(?:\.\d+)?)\s*%", resp.get('reflectie', ''), re.IGNORECASE)
                if sl_match:
                    try:
                        sl_value = float(sl_match.group(1)) / 100.0
                        if 0.001 < sl_value < 0.5: # Sanity check for SL percentage
                            ai_recommended_sl_pct = sl_value
                            highest_confidence_for_sl = resp_confidence
                            logger.debug(f"AI ({'GPT' if resp == gpt_response else 'Grok'}) raadt SL aan: {ai_recommended_sl_pct:.2%} met confidence {resp_confidence:.2f}")
                    except ValueError:
                        logger.warning(f"Kon SL percentage niet parsen uit AI response: {sl_match.group(1)}")


        if ai_recommended_sl_pct is not None:
            adjusted_trailing_offset = ai_recommended_sl_pct # Start with AI recommendation

            # Adjust based on bias and confidence (example factors)
            bias_factor = 1.0 - ((learned_bias - 0.5) * 0.2) # More bias -> tighter SL if bias is < 0.5 (bearish)
            confidence_factor = 1.0 - ((learned_confidence - 0.5) * 0.2) # More confidence -> tighter SL if conf > 0.5

            final_trailing_offset = adjusted_trailing_offset * bias_factor * confidence_factor
            final_trailing_offset = max(0.005, min(final_trailing_offset, 0.10)) # Clamp between 0.5% and 10%

            # Calculate a hard stoploss based on this (e.g., 1.5x the offset)
            hard_stoploss_value = -(final_trailing_offset * 1.5)
            hard_stoploss_value = max(-0.20, min(hard_stoploss_value, -0.01)) # Clamp hard SL

            # Dynamic trailing stop trigger based on profit (example)
            trailing_stop_trigger = 0.005 # Default trigger
            current_profit_pct = trade.get('profit_pct', 0.0)
            if current_profit_pct > 0.05: # If profit is > 5%
                trailing_stop_trigger = current_profit_pct * 0.5 # e.g., trigger at 50% of current profit

            logger.info(f"[ExitOptimizer] AI SL Optimalisatie voor {symbol}: "
                        f"Hard SL: {hard_stoploss_value:.2%}, "
                        f"Trailing Trigger: {trailing_stop_trigger:.2%}, "
                        f"Trailing Offset: {final_trailing_offset:.2%}")

            return {
                "stoploss": hard_stoploss_value,
                "trailing_stop_positive_offset": final_trailing_offset,
                "trailing_stop_positive": trailing_stop_trigger
            }

        logger.debug(f"[ExitOptimizer] Geen AI-advies voor SL-optimalisatie voor {symbol} of confidence te laag.")
        return None

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
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
            return {"patterns": {}} # Default no patterns
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_patterns

        optimizer.bias_reflector.get_bias_score = lambda t, s: 0.4
        if optimizer.confidence_engine:
            optimizer.confidence_engine.get_confidence_score = lambda t, s: 0.6

        # Mock params_manager.get_param to return specific values for keys
        def mock_get_param(key, strategy_id=None):
            if key == "exitConvictionDropTrigger": return 0.4
            if key == "cnnPatternWeight": return 1.0
            return None # Default for other keys
        optimizer.params_manager.get_param = mock_get_param


        # --- Test should_exit ---
        print("\n--- Test ExitOptimizer (should_exit) ---")

        # Scenario 1: Strong Bearish Pattern (weighted score) leads to exit
        print("\n--- Test Scenario 1: Strong Bearish Pattern (Weighted) ---")
        async def mock_ask_ai_hold_high_conf(*args, **kwargs): # AI is confident to hold, but pattern should override
            return {"reflectie": "Alles ziet er goed uit, HODL!", "confidence": 0.8, "intentie": "HOLD", "emotie": "optimistisch"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_hold_high_conf
        optimizer.grok_reflector.ask_grok = mock_ask_ai_hold_high_conf

        async def mock_cnn_strong_bearish_rule(*args, **kwargs):
            return {"patterns": {"eveningStar": True}} # Rule-based bearish
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_strong_bearish_rule

        # Params: cnnPatternWeight = 1.0 (from default mock_get_param)
        # weighted_pattern_score = 0.7 (from rule) * 1.0 = 0.7
        # strong_bearish_pattern_threshold = 0.5. So, 0.7 >= 0.5 is True.
        # AI confidence (0.8) > 0.5. So, exit.
        exit_decision_s1 = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.5, learned_confidence=0.8, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print(f"Resultaat (Scenario 1 - Strong Bearish Pattern): {json.dumps(exit_decision_s1, indent=2, default=str)}")
        assert exit_decision_s1['exit'] is True
        assert exit_decision_s1['reason'] == "bearish_pattern_with_ai_confirmation"
        assert exit_decision_s1['weighted_bearish_score'] == 0.7

        # Scenario 2: AI low confidence while in profit (original test, should still work)
        print("\n--- Test Scenario 2: Low AI Confidence in Profit ---")
        async def mock_ask_ai_low_conf(*args, **kwargs):
            # await asyncio.sleep(0.01)
            return {"reflectie": "Onzeker beeld.", "confidence": 0.3, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_low_conf
        optimizer.grok_reflector.ask_grok = mock_ask_ai_low_conf
        async def mock_cnn_no_patterns(*args, **kwargs): return {"patterns": {}} # No patterns
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_no_patterns # Reset to no patterns

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

        # Scenario 3: No strong signal to exit (weighted score too low)
        print("\n--- Test Scenario 3: No Strong Signal (Low Weighted Score) ---")
        async def mock_ask_ai_hold(*args, **kwargs):
            # await asyncio.sleep(0.01)
            return {"reflectie": "Markt stabiel.", "confidence": 0.7, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_hold
        optimizer.grok_reflector.ask_grok = mock_ask_ai_hold
        async def mock_cnn_weak_bearish_ml(*args, **kwargs): # Simulate a weak ML bearish score if available
             # For now, cnn_patterns.py doesn't give bearish ML score, so this will be 0 from ML.
             # Let's assume no rule-based patterns either for this specific test.
            return {"patterns": {}}
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_weak_bearish_ml

        # Params: cnnPatternWeight = 1.0.
        # weighted_pattern_score = 0.0 (no ML bearish, no rule-based).
        # is_strong_bearish_pattern = False.
        # AI confidence 0.7, intent HOLD. Not low_ai_confidence_profit_taking as profit is negative.
        # No exit.
        exit_decision_s3 = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_losing, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print(f"Resultaat (Scenario 3 - No Strong Signal): {json.dumps(exit_decision_s3, indent=2, default=str)}")
        assert exit_decision_s3['exit'] is False
        assert exit_decision_s3['reason'] == "no_ai_exit_signal"
        assert exit_decision_s3['weighted_bearish_score'] == 0.0


        # --- Test optimize_trailing_stop_loss ---
        print("\n--- Test ExitOptimizer (optimize_trailing_stop_loss) ---")
        # Scenario: AI recommends a specific SL percentage
        async def mock_ask_ai_sl_recommendation(*args, **kwargs):
            # await asyncio.sleep(0.01)
            return {"reflectie": "Adviseer stop loss op 2.5% voor deze trade. Trailing stop trigger at 0.5%, offset 1%.", "confidence": 0.8, "intentie": "HOLD", "emotie": "voorzichtig"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_sl_recommendation
        optimizer.grok_reflector.ask_grok = mock_ask_ai_sl_recommendation # Use same mock for Grok

        sl_optimization_result = await optimizer.optimize_trailing_stop_loss(
            dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, candles_by_timeframe=mock_candles_by_timeframe
        )
        print("SL Optimalisatie Resultaat:", json.dumps(sl_optimization_result, indent=2, default=str))
        assert sl_optimization_result is not None
        assert 'stoploss' in sl_optimization_result
        assert 'trailing_stop_positive_offset' in sl_optimization_result
        assert 'trailing_stop_positive' in sl_optimization_result

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
