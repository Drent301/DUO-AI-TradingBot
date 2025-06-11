# core/entry_decider.py
import logging
from typing import Dict, Any, Optional
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone # Import timezone
import os
import dotenv
import json
import sys # Import sys for stdout logging in main

# Importeer AI-modules en andere core componenten
from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.cnn_patterns import CNNPatterns
from core.params_manager import ParamsManager # NIEUW
from core.cooldown_tracker import CooldownTracker # NIEUW

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Logging level wordt nu idealiter globaal beheerd

class EntryDecider:
    """
    Neemt AI-gestuurde entry-besluiten op basis van consensus, confidence, bias, patronen,
    dynamische parameters en cooldowns.
    """

    def __init__(self, params_manager: Optional[ParamsManager] = None): # Voeg ParamsManager toe
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
        # [cite_start] Nieuwe initialisaties [cite: 133, 206, 275, 352, 422, 495]
        self.cooldown_tracker = CooldownTracker() # NIEUW
        self.params_manager = params_manager if params_manager else ParamsManager() # NIEUW
        logger.info("EntryDecider geïnitialiseerd met alle componenten.")


    async def get_consensus(self, prompt: str, token: str, strategy_id: str, current_learned_bias: float, current_learned_confidence: float) -> Dict[str, Any]:
        # ... (vorige implementatie van get_consensus - geen grote wijzigingen hier nodig voor deze update) ...
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
            consensus_intentie = "HOLD"
            logger.info(f"Conflicting AI intentions for {token}: GPT={gpt_intentie}, Grok={grok_intentie}. Defaulting to HOLD.")

        num_valid_confidences = sum(1 for conf in [gpt_confidence, grok_confidence] if isinstance(conf, (float, int)) and conf >= 0)
        combined_confidence = (gpt_confidence + grok_confidence) / num_valid_confidences if num_valid_confidences > 0 else 0.0

        gpt_reported_bias = gpt_response.get('bias', current_learned_bias)
        if not isinstance(gpt_reported_bias, (float, int)): gpt_reported_bias = current_learned_bias
        grok_reported_bias = grok_response.get('bias', current_learned_bias)
        if not isinstance(grok_reported_bias, (float, int)): grok_reported_bias = current_learned_bias
        combined_bias_reported = (gpt_reported_bias + grok_reported_bias) / 2.0

        logger.info(f"AI Consensus voor {token}: Intentie={consensus_intentie}, Confidence={combined_confidence:.2f}, Reported Bias={combined_bias_reported:.2f}")
        return {
            "consensus_intentie": consensus_intentie,
            "combined_confidence": combined_confidence,
            "combined_bias_reported": combined_bias_reported,
            "gpt_raw": gpt_response,
            "grok_raw": grok_response
        }

    async def should_enter(
        self,
        dataframe: pd.DataFrame,
        symbol: str,
        current_strategy_id: str,
        trade_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        logger.debug(f"[EntryDecider] Evalueren entry voor {symbol} met strategie {current_strategy_id}...")

        # [cite_start] Cooldown check (NIEUW) [cite: 63, 133, 206, 249, 275, 326, 352, 396, 422, 471, 495, 564]
        action_key_entry = f"entry_{symbol}" # Of specifieker indien nodig
        if await self.cooldown_tracker.is_cooldown_active(symbol, current_strategy_id, action_key_entry):
            cooldown_details = await self.cooldown_tracker.get_cooldown_details(symbol, current_strategy_id, action_key_entry)
            logger.info(f"[EntryDecider] Entry voor {symbol} in cooldown. Details: {cooldown_details}")
            return {"enter": False, "reason": "cooldown_active", "cooldown_details": cooldown_details, "ai_intent": "HOLD", "pattern_details": {}}


        if dataframe.empty:
            logger.warning(f"[EntryDecider] Geen dataframe beschikbaar voor {symbol}. Kan geen entry besluit nemen.")
            return {"enter": False, "reason": "no_dataframe", "confidence": 0, "learned_bias": 0.5, "ai_intent": "HOLD", "pattern_details": {}}

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
                prompt_type='marketAnalysis',
                current_bias=learned_bias,
                current_confidence=learned_confidence,
                additional_context=trade_context # trade_context kan hier ook meegegeven worden
            )
        else:
            logger.error("PromptBuilder not available. Cannot generate prompt.") # Zou niet mogen gebeuren als init goed gaat
            prompt = f"Analyseer de huidige markt voor {symbol} voor een LONG entry. Geleerde bias: {learned_bias:.2f}, geleerde confidence: {learned_confidence:.2f}."


        if not prompt: # Dubbelcheck, hoewel de fallback hierboven dit zou moeten voorkomen
            logger.warning(f"[EntryDecider] Geen prompt gegenereerd voor {symbol}. Entry geweigerd.")
            return {"enter": False, "reason": "no_prompt", "confidence": learned_confidence, "learned_bias": learned_bias, "ai_intent": "HOLD", "pattern_details": {}}

        consensus_result = await self.get_consensus(prompt, symbol, current_strategy_id, learned_bias, learned_confidence)
        consensus_intentie = consensus_result['consensus_intentie']
        ai_combined_confidence = consensus_result['combined_confidence']

        # [cite_start] Dynamische drempelwaarden via ParamsManager (NIEUW) [cite: 65, 135, 209, 277, 354, 424, 498, 566]
        # [cite_start] entry_conviction_threshold uit manifest [cite: 65, 135, 209, 277, 354, 424, 498, 566]
        entry_conviction_threshold = self.params_manager.get_param_value(
            'entry_conviction_threshold', strategy_id=current_strategy_id, symbol=symbol, default=0.7
        )
        # [cite_start] bias_threshold_for_entry uit manifest (niet direct gevonden, maar logisch)
        bias_threshold_for_entry = self.params_manager.get_param_value(
            'bias_threshold_for_entry', strategy_id=current_strategy_id, symbol=symbol, default=0.55
        )
        # [cite_start] cnn_pattern_strength_threshold uit manifest (nieuw, voor patroongewicht) [cite: 54, 124, 198, 267, 343, 413, 487, 555]
        # Dit is een placeholder, de daadwerkelijke implementatie van patroongewicht is complexer
        cnn_pattern_strength_threshold = self.params_manager.get_param_value(
            'cnn_pattern_strength_threshold', strategy_id=current_strategy_id, symbol=symbol, default=0.6 # Arbitraire default
        )


        pattern_data = await self.cnn_patterns_detector.detect_patterns_multi_timeframe(candles_by_timeframe, symbol)
        has_strong_cnn_pattern = False
        pattern_details_for_log = {}

        if pattern_data and pattern_data.get('patterns'):
            pattern_details_for_log = pattern_data['patterns']
            bullish_patterns = ['bullishEngulfing', 'CDLENGULFING', 'morningStar', 'CDLMORNINGSTAR',
                                'threeWhiteSoldiers', 'CDL3WHITESOLDIERS', 'bullFlag', 'bullishFractal',
                                'bullishRSIDivergence', 'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLPIERCING', 'ascendingTriangle', 'pennant']

            # Check voor een sterk bullish patroon, eventueel gewogen door cnn_pattern_strength_threshold
            # Voor nu, simpele check of een van de patronen aanwezig is en een mock 'strength' heeft.
            # In een echte implementatie zou 'strength' uit CNN komen of via ParamsManager per patroon.
            for p_name, p_details in pattern_data['patterns'].items():
                # Stel dat p_details een dict is met {'strength': float} of gewoon True
                p_strength = p_details if isinstance(p_details, float) else (p_details.get('strength', 1.0) if isinstance(p_details, dict) else 1.0)
                if any(b_p.upper() == p_name.upper() for b_p in bullish_patterns) and p_strength >= cnn_pattern_strength_threshold:
                    has_strong_cnn_pattern = True
                    logger.info(f"Sterk bullish CNN patroon ({p_name}, strength {p_strength:.2f}) gedetecteerd voor {symbol}")
                    break

        # [cite_start] cooldownDuration uit manifest [cite: 63, 133, 206, 249, 275, 326, 352, 396, 422, 471, 495, 564]
        # Cooldown is al gecheckt aan het begin van de methode. Hier zetten we het als de entry succesvol is.
        cooldown_duration_entry = self.params_manager.get_param_value(
            'cooldownDuration', strategy_id=current_strategy_id, symbol=symbol, param_group='entry', default=30
        )


        if consensus_intentie == "LONG" and \
            ai_combined_confidence >= entry_conviction_threshold and \
            learned_bias >= bias_threshold_for_entry and \
            has_strong_cnn_pattern:

            logger.info(f"[EntryDecider] ✅ Entry GOEDKEURING voor {symbol}. Consensus: {consensus_intentie}, AI Conf: {ai_combined_confidence:.2f}, Geleerde Bias: {learned_bias:.2f}, Patroon: {has_strong_cnn_pattern}.")

            # [cite_start] Zet cooldown na succesvolle entry (NIEUW) [cite: 63, 133, 206, 249, 275, 326, 352, 396, 422, 471, 495, 564]
            await self.cooldown_tracker.set_cooldown(symbol, current_strategy_id, action_key_entry, minutes=cooldown_duration_entry)

            return {
                "enter": True, "reason": "AI_CONSENSUS_LONG",
                "confidence": ai_combined_confidence, "learned_bias": learned_bias,
                "ai_intent": consensus_intentie, "ai_details": consensus_result,
                "pattern_details": pattern_details_for_log
            }
        else:
            reason_parts = []
            if consensus_intentie != "LONG": reason_parts.append(f"ai_intent_{consensus_intentie}")
            if ai_combined_confidence < entry_conviction_threshold: reason_parts.append(f"ai_conf_low_{ai_combined_confidence:.2f}<{entry_conviction_threshold:.2f}")
            if learned_bias < bias_threshold_for_entry: reason_parts.append(f"learned_bias_low_{learned_bias:.2f}<{bias_threshold_for_entry:.2f}")
            if not has_strong_cnn_pattern: reason_parts.append("no_strong_bullish_pattern")
            # Cooldown check is aan het begin, dus hier niet meer nodig als reden voor weigering.

            full_reason_str = "_".join(reason_parts) if reason_parts else "conditions_not_met"
            logger.info(f"[EntryDecider] ❌ Entry GEWEIGERD voor {symbol}. Reden: {full_reason_str}. AI Intentie: {consensus_intentie}, AI Conf: {ai_combined_confidence:.2f}, Geleerde Bias: {learned_bias:.2f}.")
            return {
                "enter": False, "reason": full_reason_str,
                "confidence": ai_combined_confidence, "learned_bias": learned_bias,
                "ai_intent": consensus_intentie, "ai_details": consensus_result,
                "pattern_details": pattern_details_for_log
            }

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.DEBUG, # DEBUG om alle logs te zien
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if not os.path.exists(dotenv_path):
        logger.warning(f".env file not found at {dotenv_path}. API keys might be missing.")
    dotenv.load_dotenv(dotenv_path)

    # Zet dummy API keys als ze niet in .env staan, voor testdoeleinden
    if not os.getenv("OPENAI_API_KEY"): os.environ["OPENAI_API_KEY"] = "dummy_openai_key_for_testing"
    if not os.getenv("GROK_API_KEY"): os.environ["GROK_API_KEY"] = "dummy_grok_key_for_testing"


    # Mock Freqtrade DataFrame
    def create_mock_dataframe_for_entry(timeframe: str = '5m', num_candles: int = 100) -> pd.DataFrame:
        # ... (vorige mock dataframe implementatie) ...
        data = []
        now = datetime.utcnow()
        interval_seconds_map = { '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400 }
        interval_seconds = interval_seconds_map.get(timeframe, 300)
        for i in range(num_candles):
            date = now - timedelta(seconds=(num_candles - 1 - i) * interval_seconds)
            open_ = 100 + i * 0.1 + np.random.rand() * 2; close_ = open_ + (np.random.rand() - 0.5) * 5
            high_ = max(open_, close_) + np.random.rand() * 2; low_ = min(open_, close_) - np.random.rand() * 2
            volume = 1000 + np.random.rand() * 500; rsi = 50 + (np.random.rand() - 0.5) * 30
            macd_val = (np.random.rand() - 0.5) * 0.1; macdsignal_val = macd_val * (0.8 + np.random.rand() * 0.2)
            macdhist_val = macd_val - macdsignal_val; sma_period = 20; std_dev_multiplier = 2
            if i >= sma_period -1 and len(data) >= sma_period -1:
                recent_closes_for_bb = [r[4] for r in data[-(sma_period-1):]] + [close_]
                if len(recent_closes_for_bb) == sma_period:
                    sma = np.mean(recent_closes_for_bb); std_dev = np.std(recent_closes_for_bb)
                    bb_middle = sma; bb_upper = sma + std_dev_multiplier * std_dev; bb_lower = sma - std_dev_multiplier * std_dev
                else: bb_middle=open_; bb_upper=high_; bb_lower=low_
            else: bb_middle=open_; bb_upper=high_; bb_lower=low_
            data.append([date, open_, high_, low_, close_, volume, rsi, macd_val, macdsignal_val, macdhist_val, bb_upper, bb_middle, bb_lower])
        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_upperband', 'bb_middleband', 'bb_lowerband'])
        df['date'] = pd.to_datetime(df['date']); df.set_index('date', inplace=True)
        df.attrs['timeframe'] = timeframe; df.attrs['pair'] = 'ETH/USDT'
        return df

    async def run_test_entry_decider():
        # Initialiseer ParamsManager (kan leeg zijn voor test of gevuld met mock params)
        mock_params_manager = ParamsManager()
        # Optioneel: voeg test parameters toe aan mock_params_manager als nodig
        # mock_params_manager.set_param_value('entry_conviction_threshold', 'DUOAI_Strategy', 'ETH/USDT', 0.65)

        decider = EntryDecider(params_manager=mock_params_manager) # Geef ParamsManager mee
        test_symbol = "ETH/USDT"
        test_strategy_id = "DUOAI_Strategy"

        mock_df = create_mock_dataframe_for_entry(timeframe='5m', num_candles=60)
        last_idx = mock_df.index[-1]
        mock_df.loc[last_idx, 'open'] = mock_df.loc[last_idx, 'low'] + 0.1
        mock_df.loc[last_idx, 'close'] = mock_df.loc[last_idx, 'high'] - 0.1
        mock_df.loc[last_idx, 'volume'] = mock_df['volume'].mean() * 3

        original_prompt_builder_generate = decider.prompt_builder.generate_prompt_with_data if decider.prompt_builder else None
        original_gpt_ask = decider.gpt_reflector.ask_ai
        original_grok_ask = decider.grok_reflector.ask_grok
        original_cnn_detect = decider.cnn_patterns_detector.detect_patterns_multi_timeframe
        original_bias_get = decider.bias_reflector.get_bias_score
        original_conf_get = decider.confidence_engine.get_confidence_score if decider.confidence_engine else None
        original_cooldown_is_active = decider.cooldown_tracker.is_cooldown_active # NIEUW
        original_cooldown_set = decider.cooldown_tracker.set_cooldown       # NIEUW
        original_cooldown_get_details = decider.cooldown_tracker.get_cooldown_details # NIEUW


        # --- Scenario 1: Positieve Entry (met cooldown check) ---
        print("\n--- Test EntryDecider (Positief Scenario + Cooldown) ---")
        if decider.prompt_builder:
            async def mock_generate_prompt_positive(**kwargs): await asyncio.sleep(0.01); return "Mock prompt for positive entry."
            decider.prompt_builder.generate_prompt_with_data = mock_generate_prompt_positive
        async def mock_ask_gpt_positive(prompt, context): await asyncio.sleep(0.01); return {"reflectie": "GPT: Markt veelbelovend.", "confidence": 0.85, "intentie": "LONG", "bias": 0.7}
        decider.gpt_reflector.ask_ai = mock_ask_gpt_positive
        async def mock_ask_grok_positive(prompt, context): await asyncio.sleep(0.01); return {"reflectie": "Grok: Sterke accumulatie.", "confidence": 0.8, "intentie": "LONG", "bias": 0.65}
        decider.grok_reflector.ask_grok = mock_ask_grok_positive
        async def mock_detect_cnn_positive(cbt, s): await asyncio.sleep(0.01); return {"patterns": {"bullishEngulfing": {"strength": 0.8}}} # Patroon met voldoende strength
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_positive
        decider.bias_reflector.get_bias_score = lambda t, s: 0.7
        if decider.confidence_engine: decider.confidence_engine.get_confidence_score = lambda t, s: 0.8

        # Mock cooldown: Eerst geen cooldown
        async def mock_cooldown_is_active_false(*args, **kwargs): await asyncio.sleep(0.01); return False
        decider.cooldown_tracker.is_cooldown_active = mock_cooldown_is_active_false
        async def mock_cooldown_set(*args, **kwargs): await asyncio.sleep(0.01); logger.debug(f"Mock Cooldown SET aangeroepen: {args}, {kwargs}") # Log de call
        decider.cooldown_tracker.set_cooldown = mock_cooldown_set

        entry_decision = await decider.should_enter(mock_df, test_symbol, test_strategy_id, trade_context={"stake_amount": 100})
        print("\nResultaat Entry Besluit (Positief Scenario):", json.dumps(entry_decision, indent=2, default=str))
        assert entry_decision['enter'] is True, f"Positief scenario faalde, reden: {entry_decision.get('reason')}"
        assert entry_decision['reason'] == "AI_CONSENSUS_LONG"

        # --- Scenario 1b: Cooldown Actief ---
        print("\n--- Test EntryDecider (Cooldown Actief) ---")
        async def mock_cooldown_is_active_true(*args, **kwargs): await asyncio.sleep(0.01); return True # Nu is cooldown actief
        decider.cooldown_tracker.is_cooldown_active = mock_cooldown_is_active_true
        async def mock_cooldown_get_details_active(*args, **kwargs): await asyncio.sleep(0.01); return {"cooldown_until": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(), "duration_minutes": 30}
        decider.cooldown_tracker.get_cooldown_details = mock_cooldown_get_details_active


        entry_decision_cooldown = await decider.should_enter(mock_df, test_symbol, test_strategy_id)
        print("\nResultaat Entry Besluit (Cooldown Actief):", json.dumps(entry_decision_cooldown, indent=2, default=str))
        assert entry_decision_cooldown['enter'] is False
        assert entry_decision_cooldown['reason'] == "cooldown_active"


        # Herstel originele cooldown mocks voor volgende tests
        decider.cooldown_tracker.is_cooldown_active = mock_cooldown_is_active_false # Zet terug naar geen cooldown voor andere tests

        # --- Scenario 2: Lage AI confidence ---
        # ... (andere scenarios blijven grotendeels hetzelfde, maar zorg dat cooldown niet interfereert)
        print("\n--- Test EntryDecider (Lage AI Confidence) ---")
        if decider.prompt_builder:
            async def mock_generate_prompt_low_confidence(**kwargs): await asyncio.sleep(0.01); return "Mock prompt for low confidence."
            decider.prompt_builder.generate_prompt_with_data = mock_generate_prompt_low_confidence
        async def mock_ask_gpt_low_conf(prompt, context): await asyncio.sleep(0.01); return {"reflectie": "GPT: Onzeker.", "confidence": 0.4, "intentie": "LONG", "bias": 0.5}
        decider.gpt_reflector.ask_ai = mock_ask_gpt_low_conf
        async def mock_ask_grok_low_conf(prompt, context): await asyncio.sleep(0.01); return {"reflectie": "Grok: Geen duidelijk signaal.", "confidence": 0.3, "intentie": "HOLD", "bias": 0.5}
        decider.grok_reflector.ask_grok = mock_ask_grok_low_conf
        # Behoud CNN mock van positief scenario (met sterk patroon)
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = mock_detect_cnn_positive
        decider.bias_reflector.get_bias_score = lambda t, s: 0.7
        if decider.confidence_engine: decider.confidence_engine.get_confidence_score = lambda t, s: 0.8

        entry_decision_low_conf = await decider.should_enter(mock_df, test_symbol, test_strategy_id)
        print("\nResultaat Entry Besluit (Lage AI Confidence):", json.dumps(entry_decision_low_conf, indent=2, default=str))
        assert entry_decision_low_conf['enter'] is False
        # De reden string bevat nu de drempelwaarde, dus pas assert aan
        assert f"ai_conf_low_0.35<0.70" in entry_decision_low_conf['reason'] or f"ai_conf_low_0.35<0.70" in entry_decision_low_conf['reason'] # Check voor float afronding

        # Restore original methods
        if decider.prompt_builder: decider.prompt_builder.generate_prompt_with_data = original_prompt_builder_generate
        decider.gpt_reflector.ask_ai = original_gpt_ask
        decider.grok_reflector.ask_grok = original_grok_ask # Corrected from ask_ai to ask_grok
        decider.cnn_patterns_detector.detect_patterns_multi_timeframe = original_cnn_detect
        decider.bias_reflector.get_bias_score = original_bias_get
        if decider.confidence_engine: decider.confidence_engine.get_confidence_score = original_conf_get
        decider.cooldown_tracker.is_cooldown_active = original_cooldown_is_active # NIEUW
        decider.cooldown_tracker.set_cooldown = original_cooldown_set             # NIEUW
        decider.cooldown_tracker.get_cooldown_details = original_cooldown_get_details # NIEUW

        logger.info("Alle EntryDecider tests voltooid.")

    asyncio.run(run_test_entry_decider())
