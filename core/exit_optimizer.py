# core/exit_optimizer.py
import logging
from typing import Dict, Any, Optional
import asyncio # Nodig voor de async main-test
import pandas as pd # Nodig voor DataFrame type hinting
import numpy as np # Voor mock data generatie in test
from datetime import datetime, timedelta # Voor mock data generatie in test
import os # Voor dotenv in test
import dotenv # Voor dotenv in test
import re # Voor SL parsing in test/example
import json # Toegevoegd voor json.dumps in test


# Importeer AI-modules
from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.cnn_patterns import CNNPatterns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from core.params_manager import ParamsManager # Import ParamsManager

class ExitOptimizer:
    """
    Biedt AI-gestuurde logica voor exitbesluiten, inclusief dynamische aanpassing
    van Freqtrade's trailing stop en dynamic stop loss.
    Vertaald van logica in exitManager.js en exitReflector.js, en manifesten.
    """

    def __init__(self, params_manager: Optional[ParamsManager] = None): # Add ParamsManager
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
        self.params_manager = params_manager if params_manager else ParamsManager() # Store/create ParamsManager
        logger.info("ExitOptimizer geïnitialiseerd.")

    async def should_exit(
        self,
        dataframe: pd.DataFrame,
        trade: Dict[str, Any], # Freqtrade trade object of vergelijkbaar
        symbol: str,
        current_strategy_id: str
    ) -> Dict[str, Any]:
        """
        Bepaalt of een exit moet worden geplaatst op basis van AI-consensus en drempelwaarden.
        """
        logger.debug(f"[ExitOptimizer] Evalueren exit voor {symbol} (trade_id: {trade.get('id', 'N/A')})...")

        if dataframe.empty:
            logger.warning(f"[ExitOptimizer] Geen dataframe beschikbaar voor {symbol}. Kan geen exit besluit nemen.")
            return {"exit": False, "reason": "no_dataframe", "confidence": 0.0, "pattern_details": {}}

        # Haal huidige geleerde bias en confidence op
        learned_bias = self.bias_reflector.get_bias_score(symbol, current_strategy_id)
        learned_confidence = self.confidence_engine.get_confidence_score(symbol, current_strategy_id) if self.confidence_engine else 0.5


        # Genereer prompt voor AI
        tf_attr = dataframe.attrs.get('timeframe', '5m')
        if not tf_attr: tf_attr = '5m'
        candles_by_timeframe = {tf_attr: dataframe}

        prompt = None
        if self.prompt_builder:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe,
                symbol=symbol,
                # [cite_start]prompt_type='riskManagement', # Or 'marketAnalysis' for exit [cite: 510]
                prompt_type='riskManagement',
                current_bias=learned_bias,
                current_confidence=learned_confidence,
                # trade_context is already passed in the call to generate_prompt_with_data
                # so it can be used within prompt_builder if needed for context
                trade_context=trade
            )

        if not prompt:
            logger.warning(f"[ExitOptimizer] Geen prompt gegenereerd voor {symbol}. Exit evaluatie kan minder accuraat zijn.")
            # Fallback prompt if builder fails
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

        # AI exit intentie: 'SELL' voor een LONG trade
        ai_exit_intent = False
        if (gpt_intentie == "SELL" and grok_intentie == "SELL") or \
            (gpt_intentie == "SELL" and grok_intentie in ["HOLD", ""] and gpt_confidence > 0.6) or \
            (grok_intentie == "SELL" and gpt_intentie in ["HOLD", ""] and grok_confidence > 0.6):
            ai_exit_intent = True


        # [cite_start]AI exit criteria: 'exitConvictionDropTrigger' [cite: 68, 138, 212, 280, 357, 427, 501]
        # [cite_start]exit_conviction_drop_threshold = 0.4 # Wanneer confidence te laag wordt → exit [cite: 68, 138, 212, 280, 357, 427, 501]
        exit_conviction_drop_threshold = 0.4

        # Check voor bearish patronen (uit cnn_patterns.py)
        pattern_data = await self.cnn_patterns_detector.detect_patterns_multi_timeframe(candles_by_timeframe, symbol)
        has_strong_bearish_pattern = False
        if pattern_data and pattern_data.get('patterns'):
            bearish_patterns = ['bearishEngulfing', 'CDLENGULFING', 'eveningStar', 'CDLEVENINGSTAR',
                                'threeBlackCrows', 'CDL3BLACKCROWS', 'darkCloudCover', 'CDLDARKCLOUDCOVER',
                                'bearishRSIDivergence', 'CDLHANGINGMAN', 'doubleTop', 'descendingTriangle', 'parabolicCurveDown']
            if any(p.upper() in (key.upper() for key in pattern_data['patterns'].keys()) for p in bearish_patterns):
                has_strong_bearish_pattern = True
                logger.info(f"Sterk bearish CNN patroon gedetecteerd voor {symbol} (exit eval): {pattern_data['patterns']}")


        # AI-besluitvormingslogica voor exit
        current_profit_pct = trade.get('profit_pct', 0.0)

        # Scenario 1: AI is onzeker en trade is in winst (take profit)
        if combined_confidence < exit_conviction_drop_threshold and current_profit_pct > 0.005: # Minimum 0.5% winst
            logger.info(f"[ExitOptimizer] ✅ Exit door lage AI confidence ({combined_confidence:.2f} < {exit_conviction_drop_threshold}) en trade in winst ({current_profit_pct:.2%}) voor {symbol}.")
            return {"exit": True, "reason": "low_ai_confidence_profit_taking", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

        # Scenario 2: Sterk bearish patroon met AI-bevestiging (zelfs als confidence niet extreem laag is)
        if has_strong_bearish_pattern and combined_confidence > 0.5: # Bearish patroon en AI is ten minste matig zeker
             logger.info(f"[ExitOptimizer] ✅ Exit door sterk bearish patroon en AI confidence {combined_confidence:.2f} voor {symbol}.")
             return {"exit": True, "reason": "bearish_pattern_with_ai_confirmation", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

        # Scenario 3: AI wil verkopen met voldoende confidence
        if ai_exit_intent and combined_confidence > 0.6: # AI wil verkopen met goede confidence
            logger.info(f"[ExitOptimizer] ✅ Exit door AI verkoop intentie (GPT: {gpt_intentie}, Grok: {grok_intentie}) met confidence {combined_confidence:.2f} voor {symbol}.")
            return {"exit": True, "reason": "ai_sell_intent_confident", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

        logger.debug(f"[ExitOptimizer] Geen AI-gedreven exit trigger voor {symbol}. AI Conf: {combined_confidence:.2f}, AI Intent GPT: {gpt_intentie}, Grok: {grok_intentie}.")
        return {"exit": False, "reason": "no_ai_exit_signal", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

    async def optimize_trailing_stop_loss(
        self,
        dataframe: pd.DataFrame,
        trade: Dict[str, Any],
        symbol: str,
        current_strategy_id: str
    ) -> Optional[Dict[str, float]]:
        """
        Optimaliseert dynamisch de trailing stop loss op basis van AI-inzichten.
        [cite_start]Gebruikt AI-afgeleide SL op basis van context. [cite: 59, 129, 202, 271, 348, 418, 491, 560]
        [cite_start][cite: 38] Freqtrade heeft trailing_stop en dynamic_stop_loss. [cite_start]We sturen deze AI-gedreven aan. [cite: 372]
        Retourneert een dictionary met Freqtrade compatibele stop loss parameters of None.
        """
        logger.debug(f"[ExitOptimizer] Optimaliseren trailing stop loss voor {symbol} (trade_id: {trade.get('id', 'N/A')})...")

        learned_bias = self.bias_reflector.get_bias_score(symbol, current_strategy_id)
        learned_confidence = self.confidence_engine.get_confidence_score(symbol, current_strategy_id) if self.confidence_engine else 0.5


        tf_attr = dataframe.attrs.get('timeframe', '5m')
        if not tf_attr: tf_attr = '5m'
        candles_by_timeframe = {tf_attr: dataframe}

        prompt = None
        if self.prompt_builder:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe,
                symbol=symbol,
                prompt_type='riskManagement',
                current_bias=learned_bias,
                current_confidence=learned_confidence,
                trade_context=trade
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
            if resp_confidence > 0.55 and resp_confidence > highest_confidence_for_sl:
                # Example parsing: search for "stop loss at X%" or "trailing stop X%"
                # Regex updated to be more flexible with separators like 'op' or ':'
                sl_match = re.search(r"(?:stop loss|stoploss|sl|trailing stop|tsl)(?:\s*(?:at|is|to|should be|of|op|[:])\s*)?(\d+(?:\.\d+)?)\s*%", resp.get('reflectie', ''), re.IGNORECASE)
                if sl_match:
                    try:
                        sl_value = float(sl_match.group(1)) / 100.0 # Convert percentage to decimal
                        if 0.001 < sl_value < 0.5: # Sanity check: SL between 0.1% and 50%
                            ai_recommended_sl_pct = sl_value
                            highest_confidence_for_sl = resp_confidence
                            logger.debug(f"AI ({'GPT' if resp == gpt_response else 'Grok'}) raadt SL aan: {ai_recommended_sl_pct:.2%} met confidence {resp_confidence:.2f}")
                    except ValueError:
                        logger.warning(f"Kon SL percentage niet parsen uit AI response: {sl_match.group(1)}")


        if ai_recommended_sl_pct is not None:
            # Freqtrade's stoploss is een negatief percentage (-0.10 for 10% below entry)
            # Trailing stop trigger ('trailing_stop_positive') is een positief percentage (vanaf welke winst begint trailing)
            # Trailing stop offset ('trailing_stop_positive_offset') is de daadwerkelijke trail (bv. 0.02 for 2% onder de piek)

            # AI-gestuurde Dynamic SL aanpassing:
            # We passen hier 'trailing_stop_positive_offset' aan.
            # De 'trailing_stop_positive' (trigger) kan een vaste waarde zijn of ook AI-gedreven.

            adjusted_trailing_offset = ai_recommended_sl_pct

            # Pas TSL offset aan op basis van huidige geleerde confidence en bias
            # Hogere confidence/positieve bias -> mogelijk een strakkere trail (kleinere offset)
            # Lagere confidence/negatieve bias -> mogelijk een ruimere trail (grotere offset)
            # Factor: 1.0 is neutraal. < 1.0 is strakker, > 1.0 is ruimer.
            bias_factor = 1.0 - ((learned_bias - 0.5) * 0.2)
            confidence_factor = 1.0 - ((learned_confidence - 0.5) * 0.2)

            final_trailing_offset = adjusted_trailing_offset * bias_factor * confidence_factor
            final_trailing_offset = max(0.005, min(final_trailing_offset, 0.10)) # Clamp between 0.5% and 10%

            # De harde stoploss kan ook worden aangepast, maar is vaak een fallback.
            hard_stoploss_value = -(final_trailing_offset * 1.5)
            hard_stoploss_value = max(-0.20, min(hard_stoploss_value, -0.01))

            trailing_stop_trigger = 0.005
            current_profit_pct = trade.get('profit_pct', 0.0)
            if current_profit_pct > 0.05:
                trailing_stop_trigger = current_profit_pct * 0.5

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


    def create_mock_dataframe_for_exit(timeframe: str = '5m', num_candles: int = 100) -> pd.DataFrame:
        data = []
        now = datetime.utcnow()
        interval_seconds_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
        interval_seconds = interval_seconds_map.get(timeframe, 300)

        for i in range(num_candles):
            date = now - timedelta(seconds=(num_candles - 1 - i) * interval_seconds)
            open_ = 100 + i * 0.1 + np.random.rand() * 2
            close_ = open_ + (np.random.rand() - 0.5) * 5
            high_ = max(open_, close_) + np.random.rand() * 2
            low_ = min(open_, close_) - np.random.rand() * 2
            volume = 1000 + np.random.rand() * 500
            rsi = 50 + (np.random.rand() - 0.5) * 30
            macd_val = (np.random.rand() - 0.5) * 0.1
            macdsignal_val = macd_val * (0.8 + np.random.rand()*0.2)
            macdhist_val = macd_val - macdsignal_val
            sma_period = 20; std_dev_multiplier = 2
            if i >= sma_period-1 and len(data) >= sma_period-1:
                recent_closes_for_bb = [r[4] for r in data[-(sma_period-1):]] + [close_]
                if len(recent_closes_for_bb) == sma_period:
                    sma = np.mean(recent_closes_for_bb); std_dev = np.std(recent_closes_for_bb)
                    bb_middle = sma; bb_upper = sma + std_dev_multiplier * std_dev; bb_lower = sma - std_dev_multiplier * std_dev
                else: bb_middle=open_; bb_upper=high_; bb_lower=low_
            else: bb_middle=open_; bb_upper=high_; bb_lower=low_
            data.append([date,open_,high_,low_,close_,volume,rsi,macd_val,macdsignal_val,macdhist_val,bb_upper,bb_middle,bb_lower])

        df = pd.DataFrame(data, columns=['date','open','high','low','close','volume','rsi','macd','macdsignal','macdhist','bb_upperband','bb_middleband','bb_lowerband'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.attrs['timeframe'] = timeframe
        df.attrs['pair'] = 'ETH/USDT'
        return df

    async def run_test_exit_optimizer():
        optimizer = ExitOptimizer()
        test_symbol = "ETH/USDT"
        test_strategy_id = "DUOAI_Strategy"
        mock_trade_profitable = {"id": "test_trade_1", "pair": test_symbol, "is_open": True, "profit_pct": 0.03, "stake_amount": 100, "open_date": datetime.utcnow() - timedelta(hours=2)}
        mock_trade_losing = {"id": "test_trade_2", "pair": test_symbol, "is_open": True, "profit_pct": -0.02, "stake_amount": 100, "open_date": datetime.utcnow() - timedelta(hours=1)}


        mock_df = create_mock_dataframe_for_exit(timeframe='5m', num_candles=60)

        # Store original methods
        original_prompt_builder_generate = optimizer.prompt_builder.generate_prompt_with_data if optimizer.prompt_builder else None
        original_gpt_ask = optimizer.gpt_reflector.ask_ai
        original_grok_ask = optimizer.grok_reflector.ask_grok
        original_cnn_detect = optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe
        original_bias_get = optimizer.bias_reflector.get_bias_score
        original_conf_get = optimizer.confidence_engine.get_confidence_score if optimizer.confidence_engine else None


        # Mock dependencies
        if optimizer.prompt_builder:
            async def mock_generate_prompt_general_fixed(**kwargs): # Use **kwargs for flexibility, ensure it's async
                await asyncio.sleep(0.01)
                return f"Mock prompt for {kwargs.get('symbol')} ({kwargs.get('prompt_type')})"
            optimizer.prompt_builder.generate_prompt_with_data = mock_generate_prompt_general_fixed
        optimizer.bias_reflector.get_bias_score = lambda t, s: 0.4 # Slightly bearish learned bias
        if optimizer.confidence_engine:
            optimizer.confidence_engine.get_confidence_score = lambda t, s: 0.6 # Moderate learned confidence

        # --- Test should_exit ---
        print("\n--- Test ExitOptimizer (should_exit) ---")
        # Scenario 1: AI suggests SELL with good confidence
        async def mock_ask_gpt_sell(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Markt keert. Verkoop nu.", "confidence": 0.75, "intentie": "SELL", "emotie": "bezorgd"}
        optimizer.gpt_reflector.ask_ai = mock_ask_gpt_sell

        async def mock_ask_grok_sell(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Markt ziet er zwak uit. Exit.", "confidence": 0.7, "intentie": "SELL", "emotie": "bezorgd"}
        optimizer.grok_reflector.ask_grok = mock_ask_grok_sell

        async def mock_detect_cnn_no_patterns_for_sell_test(cbt, s): # Renamed for clarity
            await asyncio.sleep(0.01)
            return {"patterns": {}} # No patterns for this specific test case
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_no_patterns_for_sell_test

        exit_decision_sell = await optimizer.should_exit(mock_df, mock_trade_profitable, test_symbol, test_strategy_id)
        print("Exit Besluit (AI SELL):", json.dumps(exit_decision_sell, indent=2, default=str))
        assert exit_decision_sell['exit'] is True
        assert "ai_sell_intent_confident" in exit_decision_sell['reason']

        # Scenario 2: AI low confidence while in profit
        async def mock_ask_gpt_low_conf(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Onzeker beeld.", "confidence": 0.3, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.gpt_reflector.ask_ai = mock_ask_gpt_low_conf

        async def mock_ask_grok_low_conf(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Geen duidelijk signaal.", "confidence": 0.35, "intentie": "HOLD", "emotie": "onzeker"}
        optimizer.grok_reflector.ask_grok = mock_ask_grok_low_conf

        async def mock_detect_cnn_no_patterns(cbt,s):
            await asyncio.sleep(0.01)
            return {"patterns":{}}
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_no_patterns

        exit_decision_low_conf_profit = await optimizer.should_exit(mock_df, mock_trade_profitable, test_symbol, test_strategy_id)
        print("Exit Besluit (Low AI Conf in Profit):", json.dumps(exit_decision_low_conf_profit, indent=2, default=str))
        assert exit_decision_low_conf_profit['exit'] is True
        assert "low_ai_confidence_profit_taking" in exit_decision_low_conf_profit['reason']

        # Scenario 3: No strong signal to exit (should return false)
        async def mock_ask_gpt_hold(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Markt stabiel.", "confidence": 0.7, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.gpt_reflector.ask_ai = mock_ask_gpt_hold

        async def mock_ask_grok_hold(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Geen reden tot paniek.", "confidence": 0.65, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.grok_reflector.ask_grok = mock_ask_grok_hold
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_no_patterns # Re-use no patterns mock

        exit_decision_no_signal = await optimizer.should_exit(mock_df, mock_trade_losing, test_symbol, test_strategy_id) # Losing trade, no strong exit signal
        print("Exit Besluit (No AI Exit Signal):", json.dumps(exit_decision_no_signal, indent=2, default=str))
        assert exit_decision_no_signal['exit'] is False
        assert "no_ai_exit_signal" in exit_decision_no_signal['reason']


        # --- Test optimize_trailing_stop_loss ---
        print("\n--- Test ExitOptimizer (optimize_trailing_stop_loss) ---")
        # Scenario: AI recommends a specific SL percentage
        async def mock_ask_gpt_sl(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Adviseer stop loss op 2.5% voor deze trade. Trailing stop trigger at 0.5%, offset 1%.", "confidence": 0.8, "intentie": "HOLD", "emotie": "voorzichtig"}
        optimizer.gpt_reflector.ask_ai = mock_ask_gpt_sl

        async def mock_ask_grok_sl(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Aanbevolen stoploss: 2.5%.", "confidence": 0.75, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.grok_reflector.ask_grok = mock_ask_grok_sl


        sl_optimization_result = await optimizer.optimize_trailing_stop_loss(mock_df, mock_trade_profitable, test_symbol, test_strategy_id)
        print("SL Optimalisatie Resultaat:", json.dumps(sl_optimization_result, indent=2, default=str))
        assert sl_optimization_result is not None
        assert 'stoploss' in sl_optimization_result
        assert 'trailing_stop_positive_offset' in sl_optimization_result
        assert 'trailing_stop_positive' in sl_optimization_result

        # Restore original methods
        if optimizer.prompt_builder:
            optimizer.prompt_builder.generate_prompt_with_data = original_prompt_builder_generate
        optimizer.gpt_reflector.ask_ai = original_gpt_ask
        optimizer.grok_reflector.ask_ai = original_grok_ask # Corrected: was ask_ai, should be ask_grok for grok_reflector
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = original_cnn_detect
        optimizer.bias_reflector.get_bias_score = original_bias_get
        if optimizer.confidence_engine:
            optimizer.confidence_engine.get_confidence_score = original_conf_get


    asyncio.run(run_test_exit_optimizer())
