# core/entry_decider.py
import logging
from typing import Dict, Any, Optional
import asyncio # Nodig voor de async main-test
import pandas as pd # Nodig voor DataFrame type hinting
import numpy as np # Voor mock data generatie in test
from datetime import datetime, timedelta # Voor mock data generatie in test
import os # For dotenv in test
import dotenv # For dotenv in test
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

class EntryDecider:
    """
    Neemt AI-gestuurde entry-besluiten op basis van consensus, confidence, bias en patronen.
    Vertaald van logica in entryCycle.js en concepten uit entryWatcher.js.
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
        logger.info("EntryDecider geïnitialiseerd.")

    async def get_consensus(self, prompt: str, token: str, strategy_id: str, current_learned_bias: float, current_learned_confidence: float) -> Dict[str, Any]:
        """
        Vraagt AI-modellen om input en combineert hun oordelen voor consensus.
        Vertaald van getConsensus(GPT, Grok) [DUO] uit Functies Overzicht.
        """
        logger.debug(f"Vragen om AI-consensus voor {token} ({strategy_id})...")

        # Haal GPT en Grok reflecties op
        gpt_response = await self.gpt_reflector.ask_ai(prompt, context={"token": token, "strategy_id": strategy_id, "learned_bias": current_learned_bias, "learned_confidence": current_learned_confidence})
        grok_response = await self.grok_reflector.ask_grok(prompt, context={"token": token, "strategy_id": strategy_id, "learned_bias": current_learned_bias, "learned_confidence": current_learned_confidence})

        gpt_confidence = gpt_response.get('confidence', 0.0) or 0.0
        grok_confidence = grok_response.get('confidence', 0.0) or 0.0
        gpt_intentie = str(gpt_response.get('intentie', '')).upper()
        grok_intentie = str(grok_response.get('intentie', '')).upper()


        # Bepaal consensus over intentie (LONG/SHORT/HOLD)
        consensus_intentie = "HOLD"
        # Als beide overeenkomen, of een van de twee SELL/LONG is en de ander HOLD/leeg
        if gpt_intentie == grok_intentie and gpt_intentie in ["LONG", "SHORT"]:
            consensus_intentie = gpt_intentie
        elif gpt_intentie in ["LONG", "SHORT"] and (not grok_intentie or grok_intentie == "HOLD"):
            consensus_intentie = gpt_intentie
        elif grok_intentie in ["LONG", "SHORT"] and (not gpt_intentie or gpt_intentie == "HOLD"):
            consensus_intentie = grok_intentie
        elif gpt_intentie in ["LONG", "SHORT"] and grok_intentie in ["LONG", "SHORT"] and gpt_intentie != grok_intentie:
            # Conflicting signals, default to HOLD for safety
            consensus_intentie = "HOLD"
            logger.info(f"Conflicting AI intentions for {token}: GPT={gpt_intentie}, Grok={grok_intentie}. Defaulting to HOLD.")


        # Gecombineerde confidence
        num_valid_confidences = sum(1 for conf in [gpt_confidence, grok_confidence] if isinstance(conf, (float, int)) and conf >= 0) # Check if confidence is valid number
        if num_valid_confidences > 0:
            combined_confidence = (gpt_confidence + grok_confidence) / num_valid_confidences
        else:
            combined_confidence = 0.0


        # Voor de bias, neem het gemiddelde van de gerapporteerde bias (als aanwezig) of gebruik de geleerde bias
        gpt_reported_bias = gpt_response.get('bias', current_learned_bias)
        if not isinstance(gpt_reported_bias, (float, int)): gpt_reported_bias = current_learned_bias

        grok_reported_bias = grok_response.get('bias', current_learned_bias)
        if not isinstance(grok_reported_bias, (float, int)): grok_reported_bias = current_learned_bias

        combined_bias_reported = (gpt_reported_bias + grok_reported_bias) / 2.0


        logger.info(f"AI Consensus voor {token}: Intentie={consensus_intentie}, Confidence={combined_confidence:.2f}, Reported Bias={combined_bias_reported:.2f}")

        return {
            "consensus_intentie": consensus_intentie,
            "combined_confidence": combined_confidence,
            "combined_bias_reported": combined_bias_reported, # This is AI's *perceived/suggested* bias for this situation
            "gpt_raw": gpt_response,
            "grok_raw": grok_response
        }


    async def should_enter(
        self,
        dataframe: pd.DataFrame, # De Freqtrade dataframe met indicatoren
        symbol: str,
        current_strategy_id: str,
        trade_context: Optional[Dict[str, Any]] = None # Extra context zoals stake_amount
    ) -> Dict[str, Any]:
        """
        Bepaalt of een entry moet worden geplaatst op basis van AI-consensus en drempelwaarden.
        Geoptimaliseerde entry-besluitvorming met GPT/Grok-consensus.
        Vertaald van entryCycle.js logica.
        """
        logger.debug(f"[EntryDecider] Evalueren entry voor {symbol} met strategie {current_strategy_id}...")

        if dataframe.empty:
            logger.warning(f"[EntryDecider] Geen dataframe beschikbaar voor {symbol}. Kan geen entry besluit nemen.")
            return {"enter": False, "reason": "no_dataframe", "confidence": 0, "learned_bias": 0.5, "ai_intent": "HOLD", "pattern_details": {}}

        # Haal huidige geleerde bias en confidence op
        learned_bias = self.bias_reflector.get_bias_score(symbol, current_strategy_id)
        learned_confidence = self.confidence_engine.get_confidence_score(symbol, current_strategy_id)


        # Genereer prompt voor AI
        tf_attr = dataframe.attrs.get('timeframe', '5m')
        if not tf_attr: tf_attr = '5m'
        candles_by_timeframe = {tf_attr: dataframe} # Freqtrade typically provides one TF dataframe to populate_indicators/entry/exit

        prompt = None
        if self.prompt_builder:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe,
                symbol=symbol,
                prompt_type='marketAnalysis', # Specific for entry decision
                current_bias=learned_bias,
                current_confidence=learned_confidence
            )
        else:
            logger.error("PromptBuilder not available. Cannot generate prompt.")
            prompt = None

        if not prompt:
            logger.warning(f"[EntryDecider] Geen prompt gegenereerd voor {symbol}. Entry geweigerd.")
            return {"enter": False, "reason": "no_prompt", "confidence": learned_confidence, "learned_bias": learned_bias, "ai_intent": "HOLD", "pattern_details": {}}

        # Vraag AI-consensus
        consensus_result = await self.get_consensus(prompt, symbol, current_strategy_id, learned_bias, learned_confidence)

        consensus_intentie = consensus_result['consensus_intentie']
        ai_combined_confidence = consensus_result['combined_confidence']
        # ai_reported_bias = consensus_result['combined_bias_reported'] # AI's reported bias for this prompt


        # Haal actuele drempelwaarden op (kunnen lerende variabelen zijn)
        entry_conviction_threshold = 0.7 # Minimale overtuiging nodig voor entry (uit manifest)
        # learned_bias >= 0.55 (een licht positieve geleerde bias voor 'LONG')
        bias_threshold_for_entry = 0.55 # Threshold for learned bias to be considered bullish


        # AI-cooldown check (Freqtrade heeft eigen cooldown, dit is AI-specifiek)
        # Kan gebaseerd zijn op 'cooldownDuration' leerbare variabele uit manifest
        ai_cooldown_active = False # Placeholder, implementatie in BiasReflector of aparte cooldown tracker


        # CNN patroon check (uit cnn_patterns.py)
        # `cnn_patterns.detect_patterns_multi_timeframe` returns dict
        pattern_data = await self.cnn_patterns_detector.detect_patterns_multi_timeframe(candles_by_timeframe, symbol)
        has_strong_cnn_pattern = False
        if pattern_data and pattern_data.get('patterns'):
            bullish_patterns = ['bullishEngulfing', 'CDLENGULFING', 'morningStar', 'CDLMORNINGSTAR',
                                'threeWhiteSoldiers', 'CDL3WHITESOLDIERS', 'bullFlag', 'bullishFractal',
                                'bullishRSIDivergence', 'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLPIERCING', 'ascendingTriangle', 'pennant']

            # Check for any of the bullish patterns
            if any(p.upper() in (key.upper() for key in pattern_data['patterns'].keys()) for p in bullish_patterns):
                has_strong_cnn_pattern = True
                logger.info(f"Sterk bullish CNN patroon gedetecteerd voor {symbol}: {pattern_data['patterns']}")

            # Additional check: `cnnPatternWeight` from manifest
            # This would be a weight from a learning system or a config.
            # If a pattern has a high learned weight, it might increase the overall conviction.
            # For now, it's just a boolean flag.

        # AI-besluitvormingslogica
        # Combineer de geleerde bias met de AI's huidige confidence en intentie.
        if consensus_intentie == "LONG" and \
            ai_combined_confidence >= entry_conviction_threshold and \
            learned_bias >= bias_threshold_for_entry and \
            has_strong_cnn_pattern and \
            not ai_cooldown_active:

            logger.info(f"[EntryDecider] ✅ Entry GOEDKEURING voor {symbol}. Consensus: {consensus_intentie}, AI Conf: {ai_combined_confidence:.2f}, Geleerde Bias: {learned_bias:.2f}, Patroon: {has_strong_cnn_pattern}.")
            return {
                "enter": True,
                "reason": "AI_CONSENSUS_LONG",
                "confidence": ai_combined_confidence, # AI's confidence for this specific decision
                "learned_bias": learned_bias, # The strategy's overall learned bias
                "ai_intent": consensus_intentie,
                "ai_details": consensus_result,
                "pattern_details": pattern_data.get('patterns', {})
            }
        else:
            reason_parts = []
            if consensus_intentie != "LONG": reason_parts.append(f"ai_intent_{consensus_intentie}")
            if ai_combined_confidence < entry_conviction_threshold: reason_parts.append(f"ai_conf_low_{ai_combined_confidence:.2f}")
            if learned_bias < bias_threshold_for_entry: reason_parts.append(f"learned_bias_low_{learned_bias:.2f}")
            if not has_strong_cnn_pattern: reason_parts.append("no_strong_bullish_pattern")
            if ai_cooldown_active: reason_parts.append("ai_cooldown_active")

            full_reason_str = "_".join(reason_parts) if reason_parts else "conditions_not_met"


            logger.info(f"[EntryDecider] ❌ Entry GEWEIGERD voor {symbol}. Reden: {full_reason_str}. AI Intentie: {consensus_intentie}, AI Conf: {ai_combined_confidence:.2f}, Geleerde Bias: {learned_bias:.2f}.")
            return {
                "enter": False,
                "reason": full_reason_str,
                "confidence": ai_combined_confidence,
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
    def create_mock_dataframe_for_entry(timeframe: str = '5m', num_candles: int = 100) -> pd.DataFrame:
        data = []
        now = datetime.utcnow()
        interval_seconds_map = {
            '1m': 60, '5m': 300, '15m': 900, '1h': 3600,
            '4h': 14400, '12h': 43200, '1d': 86400
        }
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
            macdsignal_val = macd_val * (0.8 + np.random.rand() * 0.2)
            macdhist_val = macd_val - macdsignal_val

            sma_period = 20
            std_dev_multiplier = 2
            current_candle_data_for_sma = []
            if i >= sma_period -1 and len(data) >= sma_period -1:
                recent_closes_for_bb = [r[4] for r in data[-(sma_period-1):]] + [close_]
                if len(recent_closes_for_bb) == sma_period:
                    sma = np.mean(recent_closes_for_bb)
                    std_dev = np.std(recent_closes_for_bb)
                    bb_middle = sma
                    bb_upper = sma + std_dev_multiplier * std_dev
                    bb_lower = sma - std_dev_multiplier * std_dev
                else:
                    bb_middle = open_
                    bb_upper = high_
                    bb_lower = low_
            else:
                bb_middle = open_
                bb_upper = high_
                bb_lower = low_

            data.append([date, open_, high_, low_, close_, volume, rsi, macd_val, macdsignal_val, macdhist_val, bb_upper, bb_middle, bb_lower])

        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_upperband', 'bb_middleband', 'bb_lowerband'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.attrs['timeframe'] = timeframe
        df.attrs['pair'] = 'ETH/USDT'
        return df

    async def run_test_entry_decider():
        decider = EntryDecider()
        test_symbol = "ETH/USDT"
        test_strategy_id = "DUOAI_Strategy"

        mock_df = create_mock_dataframe_for_entry(timeframe='5m', num_candles=60)
        # Ensure last candle is bullish for pattern detection
        last_idx = mock_df.index[-1]
        mock_df.loc[last_idx, 'open'] = mock_df.loc[last_idx, 'low'] + 0.1 # Open near low
        mock_df.loc[last_idx, 'close'] = mock_df.loc[last_idx, 'high'] - 0.1 # Close near high
        mock_df.loc[last_idx, 'volume'] = mock_df['volume'].mean() * 3

        # Store original methods to restore after test
        original_prompt_builder_generate = decider.prompt_builder.generate_prompt_with_data if decider.prompt_builder else None
        original_gpt_ask = decider.gpt_reflector.ask_ai
        original_grok_ask = decider.grok_reflector.ask_grok
        original_cnn_detect = decider.cnn_patterns_detector.detect_patterns_multi_timeframe
        original_bias_get = decider.bias_reflector.get_bias_score
        original_conf_get = decider.confidence_engine.get_confidence_score if decider.confidence_engine else None


        # --- Scenario 1: Positieve Entry ---
        print("\n--- Test EntryDecider (Positief Scenario) ---")
        # Mock API calls and dependencies for a positive entry
        if decider.prompt_builder:
            async def mock_generate_prompt_positive(**kwargs):
                await asyncio.sleep(0.01)
                return "Mock prompt for positive entry."
            decider.prompt_builder.generate_prompt_with_data = mock_generate_prompt_positive

        async def mock_ask_gpt_positive(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "GPT: Markt ziet er veelbelovend uit.", "confidence": 0.85, "intentie": "LONG", "emotie": "optimistisch", "bias": 0.7}
        decider.gpt_reflector.ask_ai = mock_ask_gpt_positive

        async def mock_ask_grok_positive(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Grok: Sterke accumulatie.", "confidence": 0.8, "intentie": "LONG", "emotie": "positief", "bias": 0.65}
        decider.grok_reflector.ask_grok = mock_ask_grok_positive

        async def mock_detect_cnn_positive(cbt, s):
            await asyncio.sleep(0.01)
            return {"patterns": {"bullishEngulfing": True, "CDLENGULFING": True}, "context": {"trend": "uptrend"}}
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_positive
        decider.bias_reflector.get_bias_score = lambda t, s: 0.7 # Learned bias is positive
        if decider.confidence_engine:
            decider.confidence_engine.get_confidence_score = lambda t, s: 0.8 # Learned confidence is high


        entry_decision = await decider.should_enter(
            dataframe=mock_df, symbol=test_symbol, current_strategy_id=test_strategy_id,
            trade_context={"stake_amount": 100}
        )
        print("\nResultaat Entry Besluit (Positief Scenario):", json.dumps(entry_decision, indent=2, default=str))
        assert entry_decision['enter'] is True
        assert entry_decision['reason'] == "AI_CONSENSUS_LONG"

        # --- Scenario 2: Lage AI confidence ---
        print("\n--- Test EntryDecider (Lage AI Confidence) ---")
        if decider.prompt_builder:
            async def mock_generate_prompt_low_confidence(**kwargs):
                await asyncio.sleep(0.01)
                return "Mock prompt for low confidence entry."
            decider.prompt_builder.generate_prompt_with_data = mock_generate_prompt_low_confidence

        async def mock_ask_gpt_low_confidence(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "GPT: Onzeker.", "confidence": 0.4, "intentie": "LONG", "emotie": "neutraal", "bias": 0.5}
        decider.gpt_reflector.ask_ai = mock_ask_gpt_low_confidence

        async def mock_ask_grok_low_confidence(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Grok: Geen duidelijk signaal.", "confidence": 0.3, "intentie": "HOLD", "emotie": "onzeker", "bias": 0.5}
        decider.grok_reflector.ask_grok = mock_ask_grok_low_confidence

        async def mock_detect_cnn_low_confidence(cbt, s):
            await asyncio.sleep(0.01)
            return {"patterns": {"bullishEngulfing": True}}
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_low_confidence
        decider.bias_reflector.get_bias_score = lambda t, s: 0.7 # Learned bias still positive
        if decider.confidence_engine:
            decider.confidence_engine.get_confidence_score = lambda t, s: 0.8 # Learned confidence still high (but AI's new conf is low)


        entry_decision_low_conf = await decider.should_enter(
            dataframe=mock_df, symbol=test_symbol, current_strategy_id=test_strategy_id
        )
        print("\nResultaat Entry Besluit (Lage AI Confidence):", json.dumps(entry_decision_low_conf, indent=2, default=str))
        assert entry_decision_low_conf['enter'] is False
        assert "ai_conf_low" in entry_decision_low_conf['reason']

        # --- Scenario 3: Bearish AI intentie ---
        print("\n--- Test EntryDecider (Bearish AI Intentie) ---")
        if decider.prompt_builder:
            async def mock_generate_prompt_bearish(**kwargs):
                await asyncio.sleep(0.01)
                return "Mock prompt for bearish entry."
            decider.prompt_builder.generate_prompt_with_data = mock_generate_prompt_bearish

        async def mock_ask_gpt_bearish(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "GPT: Markt zwak, short is beter.", "confidence": 0.8, "intentie": "SHORT", "emotie": "bearish", "bias": 0.3}
        decider.gpt_reflector.ask_ai = mock_ask_gpt_bearish

        async def mock_ask_grok_bearish(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Grok: Daling verwacht.", "confidence": 0.7, "intentie": "SHORT", "emotie": "negatief", "bias": 0.25}
        decider.grok_reflector.ask_grok = mock_ask_grok_bearish

        async def mock_detect_cnn_bearish(cbt, s):
            await asyncio.sleep(0.01)
            return {"patterns": {}} # No bullish patterns
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_bearish
        decider.bias_reflector.get_bias_score = lambda t, s: 0.7 # Learned bias still positive, but AI override


        entry_decision_bearish_ai = await decider.should_enter(
            dataframe=mock_df, symbol=test_symbol, current_strategy_id=test_strategy_id
        )
        print("\nResultaat Entry Besluit (Bearish AI):", json.dumps(entry_decision_bearish_ai, indent=2, default=str))
        assert entry_decision_bearish_ai['enter'] is False
        assert "ai_intent_SHORT" in entry_decision_bearish_ai['reason']


        # --- Scenario 4: Lage Geleerde Bias ---
        print("\n--- Test EntryDecider (Lage Geleerde Bias) ---")
        decider.bias_reflector.get_bias_score = lambda t, s: 0.4 # Learned bias is low
        if decider.prompt_builder:
            async def mock_generate_prompt_low_learned_bias(**kwargs):
                await asyncio.sleep(0.01)
                return "Mock prompt for low learned bias."
            decider.prompt_builder.generate_prompt_with_data = mock_generate_prompt_low_learned_bias

        async def mock_ask_gpt_low_learned_bias(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "GPT: Markt veelbelovend.", "confidence": 0.8, "intentie": "LONG", "emotie": "optimistisch", "bias": 0.7}
        decider.gpt_reflector.ask_ai = mock_ask_gpt_low_learned_bias

        async def mock_ask_grok_low_learned_bias(prompt, context):
            await asyncio.sleep(0.01)
            return {"reflectie": "Grok: Sterke accumulatie.", "confidence": 0.75, "intentie": "LONG", "emotie": "positief", "bias": 0.65}
        decider.grok_reflector.ask_grok = mock_ask_grok_low_learned_bias

        async def mock_detect_cnn_low_learned_bias(cbt, s):
            await asyncio.sleep(0.01)
            return {"patterns": {"bullishEngulfing": True}}
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_low_learned_bias
        if decider.confidence_engine:
            decider.confidence_engine.get_confidence_score = lambda t, s: 0.8 # Learned confidence high


        entry_decision_low_learned_bias = await decider.should_enter(
            dataframe=mock_df, symbol=test_symbol, current_strategy_id=test_strategy_id
        )
        print("\nResultaat Entry Besluit (Lage Geleerde Bias):", json.dumps(entry_decision_low_learned_bias, indent=2, default=str))
        assert entry_decision_low_learned_bias['enter'] is False
        assert "learned_bias_low" in entry_decision_low_learned_bias['reason']


        # Restore original methods
        if decider.prompt_builder:
            decider.prompt_builder.generate_prompt_with_data = original_prompt_builder_generate
        decider.gpt_reflector.ask_ai = original_gpt_ask
        decider.grok_reflector.ask_ai = original_grok_ask # This was ask_grok, corrected to ask_ai
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = original_cnn_detect
        decider.bias_reflector.get_bias_score = original_bias_get
        if decider.confidence_engine:
            decider.confidence_engine.get_confidence_score = original_conf_get


    asyncio.run(run_test_entry_decider())
