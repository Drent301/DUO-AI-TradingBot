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
from core.params_manager import ParamsManager # To get exitConvictionDropTrigger

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
        # cnn_patterns_detector expects candles_by_timeframe
        pattern_data = await self.cnn_patterns_detector.detect_patterns_multi_timeframe(candles_by_timeframe, symbol)

        # Get learned weight for cnnPattern from params_manager
        cnn_pattern_weight = self.params_manager.get_param("cnnPatternWeight", strategy_id=current_strategy_id)
        if cnn_pattern_weight is None: # Check for None specifically
            cnn_pattern_weight = 0.5
            logger.debug(f"cnnPatternWeight not found for strategy {current_strategy_id}, using default: {cnn_pattern_weight}")
        elif not isinstance(cnn_pattern_weight, (float, int)):
            logger.warning(f"cnnPatternWeight is not a float or int ({type(cnn_pattern_weight)}), using default 0.5 instead.")
            cnn_pattern_weight = 0.5

        weighted_pattern_score = 0.0
        base_tf_name = dataframe.attrs.get('timeframe', '5m') # Default to 5m if not set

        if pattern_data and pattern_data.get('cnn_predictions'):
            # Example: using bearishEngulfing_score. Add other relevant bearish scores.
            bear_engulf_score = pattern_data['cnn_predictions'].get(f"{base_tf_name}_bearishEngulfing_score", 0.0)
            if isinstance(bear_engulf_score, (int, float)):
                weighted_pattern_score += bear_engulf_score * cnn_pattern_weight
                logger.debug(f"CNN BearishEngulfing score for {base_tf_name}: {bear_engulf_score:.4f}, Contributed to weighted score: {bear_engulf_score * cnn_pattern_weight:.4f}")
            # Add other specific bearish CNN score contributions here if defined, e.g. bearFlag_score
            # bear_flag_score = pattern_data['cnn_predictions'].get(f"{base_tf_name}_bearFlag_score", 0.0)
            # if isinstance(bear_flag_score, (int, float)):
            #     weighted_pattern_score += bear_flag_score * cnn_pattern_weight # Assuming bear flags also contribute positively to a "bearish score"
            #     logger.debug(f"CNN BearFlag score for {base_tf_name}: {bear_flag_score:.4f}, Contributed to weighted score: {bear_flag_score * cnn_pattern_weight:.4f}")


        rule_based_bearish_patterns = ['bearishEngulfing', 'CDLENGULFING', 'eveningStar', 'CDLEVENINGSTAR',
                                   'threeBlackCrows', 'CDL3BLACKCROWS', 'darkCloudCover', 'CDLDARKCLOUDCOVER',
                                   'bearishRSIDivergence', 'CDLHANGINGMAN', 'doubleTop', 'descendingTriangle', 'parabolicCurveDown']
        if pattern_data and pattern_data.get('patterns'):
            for pattern_name in rule_based_bearish_patterns:
                if pattern_data['patterns'].get(pattern_name, False):
                    rule_based_contribution = 0.7 * cnn_pattern_weight # As per target code example
                    weighted_pattern_score += rule_based_contribution
                    logger.debug(f"Regelgebaseerd bearish patroon {pattern_name} gedetecteerd. Contributed to weighted score: {rule_based_contribution:.4f}")
                    break # Add contribution once

        strong_bearish_pattern_threshold = 0.5 # As per target code example
        is_strong_bearish_pattern = weighted_pattern_score >= strong_bearish_pattern_threshold
        logger.info(f"Evaluatie bearish patroonsterkte voor {symbol}: Gewogen score = {weighted_pattern_score:.4f}, Drempelwaarde = {strong_bearish_pattern_threshold:.2f}. Is sterk bearish patroon: {is_strong_bearish_pattern}.")

        # AI-besluitvormingslogica voor exit
        current_profit_pct = trade.get('profit_pct', 0.0)

        # Fetch the exitConvictionDropTrigger from params_manager if not passed or to ensure it's up-to-date
        # The method signature already has a default, but strategy might pass a more current one.
        # For consistency, we could re-fetch it here, or trust the one passed from strategy.
        # The original code used the passed `exit_conviction_drop_trigger`. We will stick to that.

        # Scenario 1: AI is onzeker en trade is in winst (take profit)
        if combined_confidence < exit_conviction_drop_trigger and current_profit_pct > 0.005:
            logger.info(f"[ExitOptimizer] ✅ Exit door lage AI confidence ({combined_confidence:.2f} < {exit_conviction_drop_trigger}) en trade in winst ({current_profit_pct:.2%}) voor {symbol}.")
            return {"exit": True, "reason": "low_ai_confidence_profit_taking", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

        # Scenario 2: Sterk bearish patroon met AI-bevestiging (zelfs als confidence niet extreem laag is)
        if is_strong_bearish_pattern and combined_confidence > 0.5: # Example threshold for AI confirmation
             logger.info(f"[ExitOptimizer] ✅ Exit door sterk bearish patroon (Gewogen score: {weighted_pattern_score:.4f}) en AI confidence {combined_confidence:.2f} voor {symbol}.")
             return {"exit": True, "reason": "bearish_pattern_with_ai_confirmation", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

        # Scenario 3: AI wil verkopen met voldoende confidence
        if ai_exit_intent and combined_confidence > 0.6: # Example threshold for AI sell intent
            logger.info(f"[ExitOptimizer] ✅ Exit door AI verkoop intentie (GPT: {gpt_intentie}, Grok: {grok_intentie}) met confidence {combined_confidence:.2f} voor {symbol}.")
            return {"exit": True, "reason": "ai_sell_intent_confident", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

        logger.debug(f"[ExitOptimizer] Geen AI-gedreven exit trigger voor {symbol}. AI Conf: {combined_confidence:.2f}, AI Intent GPT: {gpt_intentie}, Grok: {grok_intentie}.")
        return {"exit": False, "reason": "no_ai_exit_signal", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

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
        def mock_get_param(key, strategy_id=None, default_value=None): # Added default_value for robustness
            params = {
                "exitConvictionDropTrigger": 0.4,
                "cnnPatternWeight": 0.8 # Using 0.8 as per issue's example for exit_optimizer tests
            }
            return params.get(key, default_value)
        optimizer.params_manager.get_param = mock_get_param

        # Define updated CNN mock functions
        async def mock_detect_strong_bearish_cnn(*args, **kwargs):
            # weighted_score = (0.9 * 0.8) + (0.7 * 0.8) = 0.72 + 0.56 = 1.28. Threshold is 0.5. Strong.
            return {"patterns": {"bearishEngulfing": True}, "cnn_predictions": {"5m_bearishEngulfing_score": 0.9}}

        async def mock_detect_weak_bearish_cnn(*args, **kwargs):
            # weighted_score = (0.1 * 0.8) = 0.08. Threshold is 0.5. Weak.
            return {"patterns": {}, "cnn_predictions": {"5m_bearishEngulfing_score": 0.1}}

        # --- Test should_exit ---
        print("\n--- Test ExitOptimizer (should_exit) ---")

        # Base AI mocks
        async def mock_ask_ai_sell_intent(*args, **kwargs):
            return {"reflectie": "Markt keert. Verkoop nu.", "confidence": 0.75, "intentie": "SELL", "emotie": "bezorgd"}

        async def mock_ask_ai_hold_intent_confident(*args, **kwargs):
            return {"reflectie": "Markt stabiel, AI houdt.", "confidence": 0.6, "intentie": "HOLD", "emotie": "neutraal"}

        async def mock_ask_ai_low_conf(*args, **kwargs):
            return {"reflectie": "Onzeker beeld.", "confidence": 0.3, "intentie": "HOLD", "emotie": "neutraal"}

        # Scenario 1: AI suggests SELL, and strong bearish pattern is present
        print("\n--- Scenario 1: AI SELL intent + Strong Bearish Pattern ---")
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_sell_intent
        optimizer.grok_reflector.ask_grok = mock_ask_ai_sell_intent
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_strong_bearish_cnn

        exit_decision_ai_sell_strong_pattern = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print("Exit Besluit (AI SELL + Strong Pattern):", json.dumps(exit_decision_ai_sell_strong_pattern, indent=2, default=str))
        assert exit_decision_ai_sell_strong_pattern['exit'] is True
        # With strong pattern score 1.28 (>=0.5) and AI confidence 0.75 (>0.5), reason should be 'bearish_pattern_with_ai_confirmation'
        # OR if AI intent 'SELL' with conf > 0.6 takes precedence, it's 'ai_sell_intent_confident'
        assert exit_decision_ai_sell_strong_pattern['reason'] in ["bearish_pattern_with_ai_confirmation", "ai_sell_intent_confident"]


        # Scenario 2: Strong Bearish Pattern (score-based) with AI HOLD (but sufficient confidence)
        print("\n--- Scenario 2: Strong Bearish Pattern + AI HOLD (sufficient confidence) ---")
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_hold_intent_confident # AI says HOLD, but confidence is 0.6
        optimizer.grok_reflector.ask_grok = mock_ask_ai_hold_intent_confident
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_strong_bearish_cnn # Pattern is strong (score 1.28)

        exit_decision_strong_pattern_ai_hold = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print("Exit Besluit (Strong Pattern + AI HOLD):", json.dumps(exit_decision_strong_pattern_ai_hold, indent=2, default=str))
        assert exit_decision_strong_pattern_ai_hold['exit'] is True
        assert exit_decision_strong_pattern_ai_hold['reason'] == "bearish_pattern_with_ai_confirmation"


        # Scenario 3: AI low confidence while in profit (pattern is weak)
        print("\n--- Scenario 3: AI Low Confidence in Profit (Weak Pattern) ---")
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_low_conf # Confidence 0.3 < exit_conviction_drop_trigger (0.4)
        optimizer.grok_reflector.ask_grok = mock_ask_ai_low_conf
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_weak_bearish_cnn # Pattern is weak (score 0.08)

        exit_decision_low_conf_profit = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_profitable, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print("Exit Besluit (Low AI Conf in Profit):", json.dumps(exit_decision_low_conf_profit, indent=2, default=str))
        assert exit_decision_low_conf_profit['exit'] is True
        assert "low_ai_confidence_profit_taking" in exit_decision_low_conf_profit['reason']


        # Scenario 4: Weak Bearish Pattern (score-based) does not trigger exit (AI HOLD)
        print("\n--- Scenario 4: Weak Bearish Pattern + AI HOLD ---")
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_hold_intent_confident # AI says HOLD, confidence 0.6
        optimizer.grok_reflector.ask_grok = mock_ask_ai_hold_intent_confident
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_weak_bearish_cnn # Pattern is weak (score 0.08)

        exit_decision_weak_pattern_ai_hold = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_losing, symbol=test_symbol, current_strategy_id=test_strategy_id, # Losing trade to not trigger profit taking
            learned_bias=0.4, learned_confidence=0.6, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print("Exit Besluit (Weak Pattern + AI HOLD):", json.dumps(exit_decision_weak_pattern_ai_hold, indent=2, default=str))
        assert exit_decision_weak_pattern_ai_hold['exit'] is False
        assert "no_ai_exit_signal" in exit_decision_weak_pattern_ai_hold['reason']


        # Scenario 5: No strong signal to exit (AI HOLD, no pattern, losing trade)
        print("\n--- Scenario 5: No AI Exit Signal (AI HOLD, No Pattern) ---")
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_hold_intent_confident
        optimizer.grok_reflector.ask_grok = mock_ask_ai_hold_intent_confident
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_patterns # Default mock (no patterns, so score 0)

        exit_decision_no_signal = await optimizer.should_exit(
            dataframe=mock_df, trade=mock_trade_losing, symbol=test_symbol, current_strategy_id=test_strategy_id,
            learned_bias=0.4, learned_confidence=0.6, exit_conviction_drop_trigger=0.4, candles_by_timeframe=mock_candles_by_timeframe
        )
        print("Exit Besluit (No AI Exit Signal):", json.dumps(exit_decision_no_signal, indent=2, default=str))
        assert exit_decision_no_signal['exit'] is False
        assert "no_ai_exit_signal" in exit_decision_no_signal['reason']


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
