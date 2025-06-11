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

class ExitOptimizer:
    """
    Biedt AI-gestuurde logica voor exitbesluiten, inclusief dynamische aanpassing
    van Freqtrade's trailing stop en dynamic stop loss.
    Vertaald van logica in exitManager.js en exitReflector.js, en manifesten.
    """

    def __init__(self):
        self.gpt_reflector = GPTReflector()
        self.grok_reflector = GrokReflector()
        try:
            self.prompt_builder = PromptBuilder()
        except Exception as e:
            logger.error(f"Failed to initialize PromptBuilder (it might be empty or have issues): {e}")
            self.prompt_builder = None # Fallback
        self.bias_reflector = BiasReflector()
        try:
            self.confidence_engine = ConfidenceEngine()
        except Exception as e:
            logger.error(f"Failed to initialize ConfidenceEngine (it might be empty or have issues): {e}")
            self.confidence_engine = None # Fallback

        self.cnn_patterns_detector = CNNPatterns()
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
            return {"exit": False, "reason": "no_dataframe", "confidence": 0.0}

        # Haal huidige bias en confidence op
        current_bias = self.bias_reflector.get_bias_score(symbol, current_strategy_id)
        if self.confidence_engine:
            current_confidence = self.confidence_engine.get_confidence_score(symbol, current_strategy_id)
        else:
            logger.warning("ConfidenceEngine not available, using default confidence 0.5 for exit decision.")
            current_confidence = 0.5


        # Genereer prompt voor AI
        tf_attr = dataframe.attrs.get('timeframe', '5m')
        if not tf_attr: tf_attr = '5m'
        candles_by_timeframe = {tf_attr: dataframe}

        prompt = None
        if self.prompt_builder:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe,
                symbol=symbol,
                prompt_type='riskManagement', # Of 'marketAnalysis' voor exit
                current_bias=current_bias,
                current_confidence=current_confidence,
                trade_context=trade # Geef trade context mee voor exit prompts
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

        num_valid_confidences = sum(1 for conf in [gpt_confidence, grok_confidence] if isinstance(conf, (float, int)) and conf > 0)
        if num_valid_confidences > 0:
            combined_confidence = (gpt_confidence + grok_confidence) / num_valid_confidences
        else:
            combined_confidence = 0.0

        # Exit intentie: SELL (of SHORT als je short selling doet en dit een long trade is)
        # Voor nu, focus op SELL als exit signaal voor een LONG trade.
        ai_exit_intent = False
        if gpt_intentie == "SELL" and grok_intentie == "SELL":
            ai_exit_intent = True
        elif gpt_intentie == "SELL" and (grok_intentie == "HOLD" or not grok_intentie) and gpt_confidence > 0.6: # GPT is more confident
            ai_exit_intent = True
        elif grok_intentie == "SELL" and (gpt_intentie == "HOLD" or not gpt_intentie) and grok_confidence > 0.6: # Grok is more confident
            ai_exit_intent = True


        # [cite_start]AI exit criteria: 'exitConvictionDropTrigger' [cite: 66]
        exit_conviction_drop_threshold = 0.4 # Voorbeeld, kan geleerd worden [cite: 66]

        # Check voor bearish patronen (uit cnn_patterns.py)
        pattern_data = await self.cnn_patterns_detector.detect_patterns_multi_timeframe(candles_by_timeframe, symbol)
        has_strong_bearish_pattern = False
        if pattern_data and pattern_data.get('patterns'):
            bearish_patterns = ['bearishEngulfing', 'CDLENGULFING', 'eveningStar', 'CDLEVENINGSTAR',
                                'threeBlackCrows', 'CDL3BLACKCROWS', 'darkCloudCover', 'CDLDARKCLOUDCOVER',
                                'bearishRSIDivergence', 'CDLHANGINGMAN', 'doubleTop']
            if any(p.upper() in (key.upper() for key in pattern_data['patterns'].keys()) for p in bearish_patterns):
                has_strong_bearish_pattern = True
                logger.info(f"Sterk bearish CNN patroon gedetecteerd voor {symbol} (exit eval): {pattern_data['patterns']}")


        # AI-besluitvormingslogica voor exit
        # Een lage AI confidence kan ook een reden zijn om te exiten, vooral als de trade in winst is.
        current_profit_pct = trade.get('profit_pct', 0.0)

        if combined_confidence < exit_conviction_drop_threshold and current_profit_pct > 0.01 : # Als in winst en AI is onzeker
            logger.info(f"[ExitOptimizer] ✅ Exit door lage AI confidence ({combined_confidence:.2f} < {exit_conviction_drop_threshold}) en trade in winst ({current_profit_pct:.2%}) voor {symbol}.")
            return {"exit": True, "reason": "low_ai_confidence_profit_taking", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

        if has_strong_bearish_pattern and combined_confidence > 0.5: # Bevestiging van patroon + redelijke confidence
             logger.info(f"[ExitOptimizer] ✅ Exit door sterk bearish patroon en AI confidence {combined_confidence:.2f} voor {symbol}.")
             return {"exit": True, "reason": "bearish_pattern_with_ai_confirmation", "confidence": combined_confidence, "pattern_details": pattern_data.get('patterns', {})}

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
        [cite_start]Gebruikt AI-afgeleide SL op basis van context[cite: 58].
        [cite_start]Freqtrade heeft trailing_stop en dynamic_stop_loss[cite: 37]. We sturen deze AI-gedreven aan.
        Retourneert een dictionary met Freqtrade compatibele stop loss parameters of None.
        """
        logger.debug(f"[ExitOptimizer] Optimaliseren trailing stop loss voor {symbol} (trade_id: {trade.get('id', 'N/A')})...")

        current_bias = self.bias_reflector.get_bias_score(symbol, current_strategy_id)
        if self.confidence_engine:
            current_confidence = self.confidence_engine.get_confidence_score(symbol, current_strategy_id)
        else:
            current_confidence = 0.5


        tf_attr = dataframe.attrs.get('timeframe', '5m')
        if not tf_attr: tf_attr = '5m'
        candles_by_timeframe = {tf_attr: dataframe}

        prompt = None
        if self.prompt_builder:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe,
                symbol=symbol,
                prompt_type='riskManagement',
                current_bias=current_bias,
                current_confidence=current_confidence,
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
            if resp_confidence > 0.55 and resp_confidence > highest_confidence_for_sl: # Only consider reasonably confident advice
                # Example parsing: search for "stop loss at X%" or "trailing stop X%"
                # This needs to be robust and match the expected AI output format
                sl_match = re.search(r"(?:stop loss|stoploss|sl|trailing stop|tsl)\s*(?:at|is|to|should be|of)?\s*(\d+(?:\.\d+)?)\s*%", resp.get('reflectie', ''), re.IGNORECASE)
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
            # Freqtrade's stoploss is een negatief percentage (-0.10 voor 10% onder entry)
            # Trailing stop trigger ('trailing_stop_positive') is een positief percentage (vanaf welke winst begint trailing)
            # Trailing stop offset ('trailing_stop_positive_offset') is de daadwerkelijke trail (bv. 0.02 voor 2% onder de piek)

            # AI-gestuurde Dynamic SL aanpassing:
            # We passen hier 'trailing_stop_positive_offset' aan.
            # De 'trailing_stop_positive' (trigger) kan een vaste waarde zijn of ook AI-gedreven.

            # Basis TSL offset van AI
            adjusted_trailing_offset = ai_recommended_sl_pct

            # Pas TSL offset aan op basis van huidige geleerde confidence en bias
            # Hogere confidence/positieve bias -> mogelijk een strakkere trail (kleinere offset)
            # Lagere confidence/negatieve bias -> mogelijk een ruimere trail (grotere offset)
            # Factor: 1.0 is neutraal. < 1.0 is strakker, > 1.0 is ruimer.
            # current_confidence (0-1), current_bias (0-1, 0.5 is neutraal)
            # Map bias from 0-1 to e.g. 0.8-1.2 for factor: (bias - 0.5) * 0.4 + 1 = (0.7-0.5)*0.4+1 = 0.08+1 = 1.08
            #                                                  (0.3-0.5)*0.4+1 = -0.08+1 = 0.92
            bias_factor = 1.0 - ((current_bias - 0.5) * 0.2) # e.g. bias 0.7 -> 0.96 (strakker), bias 0.3 -> 1.04 (ruimer)
            confidence_factor = 1.0 - ((current_confidence - 0.5) * 0.2) # e.g. conf 0.8 -> 0.94 (strakker), conf 0.2 -> 1.06 (ruimer)

            final_trailing_offset = adjusted_trailing_offset * bias_factor * confidence_factor
            final_trailing_offset = max(0.005, min(final_trailing_offset, 0.10)) # Clamp between 0.5% and 10%

            # De harde stoploss kan ook worden aangepast, maar is vaak een fallback.
            # Laten we zeggen dat de AI SL advies ook de harde SL beinvloedt.
            # Een harde SL die iets ruimer is dan de trail.
            hard_stoploss_value = -(final_trailing_offset * 1.5) # e.g., 1.5x de TSL offset
            hard_stoploss_value = max(-0.20, min(hard_stoploss_value, -0.01)) # Clamp hard SL

            # Trailing stop trigger (wanneer te activeren)
            # Kan ook AI-gedreven zijn, of een vaste waarde zoals 0.005 (0.5% winst)
            trailing_stop_trigger = 0.005
            # Als de trade al flink in de winst is, kan de trigger hoger zijn.
            current_profit_pct = trade.get('profit_pct', 0.0)
            if current_profit_pct > 0.05: # Meer dan 5% winst
                trailing_stop_trigger = current_profit_pct * 0.5 # e.g., start trailing at half the current profit

            logger.info(f"[ExitOptimizer] AI SL Optimalisatie voor {symbol}: "
                        f"Hard SL: {hard_stoploss_value:.2%}, "
                        f"Trailing Trigger: {trailing_stop_trigger:.2%}, "
                        f"Trailing Offset: {final_trailing_offset:.2%}")

            return {
                "stoploss": hard_stoploss_value,
                "trailing_stop_positive_offset": final_trailing_offset,
                "trailing_stop_positive": trailing_stop_trigger
                # Freqtrade gebruikt ook 'use_custom_stoploss': True om deze waarden te laten overrulen
            }

        logger.debug(f"[ExitOptimizer] Geen AI-advies voor SL-optimalisatie voor {symbol} of confidence te laag.")
        return None

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Corrected path for .env when running this script directly
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    # Setup basic logging for the test
    import sys
    # import json # Already imported at the top
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


    # Mock Freqtrade DataFrame (zelfde als in entry_decider)
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

        # Mock dependencies that might be empty
        if optimizer.prompt_builder is None:
            class MockPromptBuilder:
                async def generate_prompt_with_data(self, **kwargs): return f"Mock prompt for {kwargs.get('symbol','N/A')} ({kwargs.get('prompt_type','N/A')})"
            optimizer.prompt_builder = MockPromptBuilder()
        if optimizer.confidence_engine is None:
            class MockConfidenceEngine:
                def get_confidence_score(self, t, s): return 0.75 # Needs two args
            optimizer.confidence_engine = MockConfidenceEngine()

        # Patch BiasReflector and ConfidenceEngine for predictable test values
        optimizer.bias_reflector.get_bias_score = lambda t, s: 0.4 # Slightly bearish learned bias
        if optimizer.confidence_engine: # Patch only if it was initialized
            optimizer.confidence_engine.get_confidence_score = lambda t, s: 0.6 # Moderate learned confidence

        # --- Test should_exit ---
        print("\n--- Test ExitOptimizer (should_exit) ---")
        # Scenario 1: AI suggests SELL with good confidence
        async def mock_ask_ai_sell(prompt, context):
            return {"reflectie": "Markt keert. Verkoop nu.", "confidence": 0.75, "intentie": "SELL", "emotie": "bezorgd"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_sell
        optimizer.grok_reflector.ask_ai = mock_ask_ai_sell
        async def mock_cnn_bearish(candles_by_tf, symbol): return {"patterns": {"bearishEngulfing": True}}
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_cnn_bearish


        exit_decision_sell = await optimizer.should_exit(mock_df, mock_trade_profitable, test_symbol, test_strategy_id)
        print("Exit Besluit (AI SELL):", json.dumps(exit_decision_sell, indent=2, default=str))
        assert exit_decision_sell['exit'] is True
        assert "ai_sell_intent_confident" in exit_decision_sell['reason']

        # Scenario 2: AI low confidence while in profit
        async def mock_ask_ai_low_conf(prompt, context):
            return {"reflectie": "Onzeker beeld.", "confidence": 0.3, "intentie": "HOLD", "emotie": "neutraal"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_low_conf
        optimizer.grok_reflector.ask_ai = mock_ask_ai_low_conf
        optimizer.cnn_patterns_detector.detect_patterns_multi_timeframe = async def mock_cnn_neutral(c,s): return {"patterns":{}} # No strong patterns

        exit_decision_low_conf_profit = await optimizer.should_exit(mock_df, mock_trade_profitable, test_symbol, test_strategy_id)
        print("Exit Besluit (Low AI Conf in Profit):", json.dumps(exit_decision_low_conf_profit, indent=2, default=str))
        assert exit_decision_low_conf_profit['exit'] is True
        assert "low_ai_confidence_profit_taking" in exit_decision_low_conf_profit['reason']


        # --- Test optimize_trailing_stop_loss ---
        print("\n--- Test ExitOptimizer (optimize_trailing_stop_loss) ---")
        # Scenario: AI recommends a specific SL percentage
        async def mock_ask_ai_sl_recommend(prompt, context):
            return {"reflectie": "Adviseer stop loss op 2.5% voor deze trade.", "confidence": 0.8, "intentie": "HOLD", "emotie": "voorzichtig"}
        optimizer.gpt_reflector.ask_ai = mock_ask_ai_sl_recommend
        optimizer.grok_reflector.ask_ai = mock_ask_ai_sl_recommend # Both recommend for higher chance


        sl_optimization_result = await optimizer.optimize_trailing_stop_loss(mock_df, mock_trade_profitable, test_symbol, test_strategy_id)
        print("SL Optimalisatie Resultaat:", json.dumps(sl_optimization_result, indent=2, default=str))
        assert sl_optimization_result is not None
        assert 'stoploss' in sl_optimization_result
        assert 'trailing_stop_positive_offset' in sl_optimization_result


    asyncio.run(run_test_exit_optimizer())
