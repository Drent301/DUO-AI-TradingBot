# core/confidence_engine.py
import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import dotenv # Added for __main__ section

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
CONFIDENCE_MEMORY_FILE = os.path.join(MEMORY_DIR, 'confidence_memory.json')
CONFIDENCE_COOLDOWN_SECONDS = 300 # 5 minuten cooldown

os.makedirs(MEMORY_DIR, exist_ok=True)

class ConfidenceEngine:
    """
    Houdt confidence scores bij per reflectie/resultaat en leert van fout/goed scorende combinaties.
    Vertaald van confidenceAdjuster.js en confidenceCycle.js.
    """

    def __init__(self):
        self.confidence_memory = self._load_confidence_memory()
        logger.info(f"ConfidenceEngine geïnitialiseerd. Geheugen: {CONFIDENCE_MEMORY_FILE}")

    def _load_confidence_memory(self) -> Dict[str, Any]:
        try:
            if os.path.exists(CONFIDENCE_MEMORY_FILE) and os.path.getsize(CONFIDENCE_MEMORY_FILE) > 0:
                with open(CONFIDENCE_MEMORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {} # Return empty dict if file doesn't exist or is empty
        except (FileNotFoundError, json.JSONDecodeError): # FileNotFoundError for completeness
            logger.warning(f"Confidence geheugenbestand {CONFIDENCE_MEMORY_FILE} niet gevonden of corrupt. Start met leeg geheugen.")
            return {}

    def _save_confidence_memory(self):
        try:
            with open(CONFIDENCE_MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.confidence_memory, f, indent=2)
        except IOError as e:
            logger.error(f"Fout bij opslaan confidence geheugen: {e}")

    def get_confidence_score(self, token: str, strategy_id: str) -> float:
        """
        Haalt de huidige confidence score op voor een specifieke token/strategie.
        Retourneert een standaardwaarde als niet gevonden (0.5 is neutraal).
        """
        return self.confidence_memory.get(token, {}).get(strategy_id, {}).get('confidence', 0.5)

    def get_max_per_trade_pct(self, token: str, strategy_id: str) -> float:
        """
        Haalt de huidige max_per_trade_pct op voor een specifieke token/strategie.
        Deze waarde kan de stake amount beïnvloeden.
        """
        return self.confidence_memory.get(token, {}).get(strategy_id, {}).get('max_per_trade_pct', 0.1) # Default 10%


    async def update_confidence(self, token: str, strategy_id: str, ai_reported_confidence: float,
                                trade_result_pct: Optional[float] = None,
                                sentiment_data_status: Optional[str] = None,
                                first_sentiment_observed: Optional[str] = None,
                                trade_direction: Optional[str] = None):
        """
        Werkt de confidence score bij op basis van AI-reflectie en trade resultaat.
        Implementeert een cooldown.
        `ai_reported_confidence` is de confidence die de AI zelf rapporteerde (0-1).
        `trade_result_pct` is het winst/verlies percentage van een trade (e.g., 0.05 for +5%).
        """
        if token not in self.confidence_memory:
            self.confidence_memory[token] = {}
        if strategy_id not in self.confidence_memory[token]:
            # Initialize with defaults, including max_per_trade_pct
            self.confidence_memory[token][strategy_id] = {
                'confidence': 0.5,
                'last_update': None,
                'trade_count': 0,
                'total_profit_pct': 0.0,
                'max_per_trade_pct': 0.1 # Initial default max stake percentage
            }


        current_data = self.confidence_memory[token][strategy_id]
        current_learned_confidence = current_data['confidence'] # This is the 'learned' confidence
        last_update_str = current_data.get('last_update')
        last_update_time = datetime.fromisoformat(last_update_str) if last_update_str else None

        # Implementeer cooldown
        if last_update_time and (datetime.now() - last_update_time).total_seconds() < CONFIDENCE_COOLDOWN_SECONDS:
            logger.debug(f"[ConfidenceEngine] Cooldown actief voor {token}/{strategy_id}. Confidence wordt niet bijgewerkt.")
            return

        adjusted_learned_confidence = current_learned_confidence
        max_per_trade_pct = current_data.get('max_per_trade_pct', 0.1) # Huidige max_per_trade_pct

        # Logica van confidenceAdjuster.js
        # ai_reported_confidence is de directe output van een AI model voor een specifieke query.
        # current_learned_confidence is de opgeslagen, geleerde confidence over tijd.

        if trade_result_pct is not None: # Als er een trade resultaat is
            if trade_result_pct > 0: # Winst
                # Verhoog geleerde confidence, sterker als AI ook confident was in zijn (hopelijk correcte) voorspelling
                # en als de winst groot is.
                # De ai_reported_confidence is de confidence van de AI *voor* deze trade (als die beschikbaar was).
                # We nemen aan dat ai_reported_confidence de confidence was van de AI die leidde tot de trade.
                profit_boost = min(trade_result_pct * 5, 1.0) # Max boost van 1.0 bij 20% winst (0.20*5=1.0)
                confidence_increase = (ai_reported_confidence * 0.05) + (profit_boost * 0.1) # Max 0.05 + 0.1 = 0.15
                adjusted_learned_confidence += confidence_increase

                # Verhoog inzet (max_per_trade_pct) als trades goed gaan
                max_per_trade_pct = min(max_per_trade_pct + 0.01 + (profit_boost * 0.02), 0.5) # Max 50% stake
            else: # Verlies
                # Verlaag geleerde confidence, sterker als AI (te) confident was in zijn (foute) voorspelling
                # en als het verlies groot is.
                loss_penalty = min(abs(trade_result_pct) * 5, 1.0) # Max penalty van 1.0 bij 20% verlies
                confidence_decrease = (ai_reported_confidence * 0.05) + (loss_penalty * 0.1) # Max 0.05 + 0.1 = 0.15
                adjusted_learned_confidence -= confidence_decrease

                # Verlaag inzet (max_per_trade_pct) bij verlies
                max_per_trade_pct = max(max_per_trade_pct - 0.01 - (loss_penalty * 0.02), 0.02) # Min 2% stake
        else:
            # Als er geen trade resultaat is (bijv. periodieke reflectie zonder trade),
            # pas de geleerde confidence aan richting de AI's gerapporteerde confidence.
            # De 'ai_reported_confidence' is de output van de AI voor de huidige reflectie.
            # Gewogen gemiddelde:
            adjustment_weight = 0.1 # Hoeveel de nieuwe AI observatie de geleerde confidence beinvloedt
            adjusted_learned_confidence = (current_learned_confidence * (1 - adjustment_weight) + ai_reported_confidence * adjustment_weight)

        # Sentiment-Based Adjustment (only if trade_result_pct is present)
        if trade_result_pct is not None and sentiment_data_status == "present_and_not_empty" and first_sentiment_observed and trade_direction:
            sentiment_confidence_impact_on_learned_conf = 0.02
            sentiment_confidence_impact_on_stake_pct = 0.005

            sentiment_aligned_with_profit = False
            sentiment_misaligned_with_profit = False

            if trade_direction == "long":
                if first_sentiment_observed == "positive" and trade_result_pct > 0: sentiment_aligned_with_profit = True
                elif first_sentiment_observed == "negative" and trade_result_pct <= 0: sentiment_aligned_with_profit = True
                elif first_sentiment_observed == "positive" and trade_result_pct <= 0: sentiment_misaligned_with_profit = True
                elif first_sentiment_observed == "negative" and trade_result_pct > 0: sentiment_misaligned_with_profit = True
            elif trade_direction == "short":
                # Aligned for short: (Negative sent AND Profit) OR (Positive sent AND Loss)
                if (first_sentiment_observed == "negative" and trade_result_pct > 0) or \
                   (first_sentiment_observed == "positive" and trade_result_pct <= 0):
                    sentiment_aligned_with_profit = True
                # Misaligned for short: (Negative sent AND Loss) OR (Positive sent AND Profit)
                elif (first_sentiment_observed == "negative" and trade_result_pct <= 0) or \
                     (first_sentiment_observed == "positive" and trade_result_pct > 0):
                    sentiment_misaligned_with_profit = True

            if sentiment_aligned_with_profit:
                learned_conf_nudge = sentiment_confidence_impact_on_learned_conf * ai_reported_confidence
                adjusted_learned_confidence += learned_conf_nudge
                max_per_trade_pct += sentiment_confidence_impact_on_stake_pct # Corrected variable
                logger.info(f"[ConfidenceEngine] Sentiment ALIGNED ({first_sentiment_observed} for {trade_direction}, outcome {trade_result_pct:.2%}). Nudging learned_confidence by +{learned_conf_nudge:.4f}, max_per_trade_pct by +{sentiment_confidence_impact_on_stake_pct:.4f}.")
            elif sentiment_misaligned_with_profit:
                learned_conf_nudge = sentiment_confidence_impact_on_learned_conf * ai_reported_confidence
                adjusted_learned_confidence -= learned_conf_nudge
                max_per_trade_pct -= sentiment_confidence_impact_on_stake_pct # Corrected variable
                logger.info(f"[ConfidenceEngine] Sentiment MISALIGNED ({first_sentiment_observed} for {trade_direction}, outcome {trade_result_pct:.2%}). Nudging learned_confidence by -{learned_conf_nudge:.4f}, max_per_trade_pct by -{sentiment_confidence_impact_on_stake_pct:.4f}.")
            else: # Neutral sentiment or other unhandled cases
                logger.info(f"[ConfidenceEngine] Sentiment ({first_sentiment_observed}) for {trade_direction} considered neutral or no strong alignment/misalignment signal for outcome {trade_result_pct:.2%}. No additional sentiment nudge.")

        adjusted_learned_confidence = max(0.0, min(1.0, adjusted_learned_confidence)) # Houd tussen 0 en 1
        max_per_trade_pct = max(0.01, min(0.5, max_per_trade_pct)) # Clamp max_per_trade_pct (e.g., 1% to 50%)


        current_data['confidence'] = adjusted_learned_confidence
        current_data['max_per_trade_pct'] = round(max_per_trade_pct, 4) # Afronden voor consistentie
        current_data['last_update'] = datetime.now().isoformat()

        if trade_result_pct is not None: # Alleen updaten als er een trade was
            current_data['trade_count'] = current_data.get('trade_count', 0) + 1
            current_data['total_profit_pct'] = current_data.get('total_profit_pct', 0.0) + trade_result_pct

        await asyncio.to_thread(self._save_confidence_memory)
        logger.info(f"[ConfidenceEngine] Confidence voor {token}/{strategy_id} bijgewerkt naar {adjusted_learned_confidence:.3f}. MaxPerTradePct: {max_per_trade_pct:.3f}.")


# Test sectie
async def run_test_confidence_engine():
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    engine = ConfidenceEngine()

    def _reset_confidence_memory_file():
        if os.path.exists(CONFIDENCE_MEMORY_FILE):
            os.remove(CONFIDENCE_MEMORY_FILE)
        return ConfidenceEngine() # Return new instance with clean memory

    # --- Test Constants (matching those in the method) ---
    SENTIMENT_CONF_IMPACT_LEARNED = 0.02
    SENTIMENT_CONF_IMPACT_STAKE = 0.005
    DEFAULT_INITIAL_CONF = 0.5
    DEFAULT_INITIAL_STAKE_PCT = 0.1
    MIN_STAKE_PCT = 0.01
    MAX_STAKE_PCT = 0.5


    # --- Test Case 1: Initial scores ---
    engine = _reset_confidence_memory_file()
    logger.info("--- Test Case 1: Initial scores ---")
    initial_conf = engine.get_confidence_score("TEST", "strat1")
    initial_stake = engine.get_max_per_trade_pct("TEST", "strat1")
    logger.info(f"Initial learned_confidence: {initial_conf}, Initial max_per_trade_pct: {initial_stake}")
    assert initial_conf == DEFAULT_INITIAL_CONF
    assert initial_stake == DEFAULT_INITIAL_STAKE_PCT

    # --- Test Case 2: Update with profitable trade, no sentiment ---
    logger.info("\n--- Test Case 2: Profitable trade, no sentiment ---")
    # ai_reported_confidence = 0.7, trade_result_pct = 0.05 (5%)
    # profit_boost = min(0.05 * 5, 1.0) = 0.25
    # confidence_increase = (0.7 * 0.05) + (0.25 * 0.1) = 0.035 + 0.025 = 0.06
    # expected_learned_conf = 0.5 + 0.06 = 0.56
    # stake_increase = 0.01 + (0.25 * 0.02) = 0.01 + 0.005 = 0.015
    # expected_stake_pct = 0.1 + 0.015 = 0.115
    await engine.update_confidence("TEST", "strat1", ai_reported_confidence=0.7, trade_result_pct=0.05)
    conf_t2 = engine.get_confidence_score("TEST", "strat1")
    stake_t2 = engine.get_max_per_trade_pct("TEST", "strat1")
    logger.info(f"Conf: {conf_t2:.4f}, Stake: {stake_t2:.4f}")
    assert abs(conf_t2 - 0.56) < 0.0001
    assert abs(stake_t2 - 0.115) < 0.0001

    # --- Test Case 3: Losing trade, no sentiment ---
    engine = _reset_confidence_memory_file() # Reset for clean calculation
    logger.info("\n--- Test Case 3: Losing trade, no sentiment ---")
    # ai_reported_confidence = 0.6, trade_result_pct = -0.03 (-3%)
    # loss_penalty = min(abs(-0.03) * 5, 1.0) = 0.15
    # confidence_decrease = (0.6 * 0.05) + (0.15 * 0.1) = 0.03 + 0.015 = 0.045
    # expected_learned_conf = 0.5 - 0.045 = 0.455
    # stake_decrease = 0.01 + (0.15 * 0.02) = 0.01 + 0.003 = 0.013
    # expected_stake_pct = 0.1 - 0.013 = 0.087
    await engine.update_confidence("TEST", "strat2", ai_reported_confidence=0.6, trade_result_pct=-0.03)
    conf_t3 = engine.get_confidence_score("TEST", "strat2")
    stake_t3 = engine.get_max_per_trade_pct("TEST", "strat2")
    logger.info(f"Conf: {conf_t3:.4f}, Stake: {stake_t3:.4f}")
    assert abs(conf_t3 - 0.455) < 0.0001
    assert abs(stake_t3 - 0.087) < 0.0001

    # --- Test Case 4: No trade result (periodic update), no sentiment ---
    engine = _reset_confidence_memory_file()
    logger.info("\n--- Test Case 4: No trade result, no sentiment ---")
    # ai_reported_confidence = 0.8. current_learned_confidence = 0.5
    # adjustment_weight = 0.1
    # expected_learned_conf = (0.5 * (1 - 0.1) + 0.8 * 0.1) = 0.45 + 0.08 = 0.53
    # expected_stake_pct = 0.1 (should not change)
    await engine.update_confidence("TEST", "strat3", ai_reported_confidence=0.8)
    conf_t4 = engine.get_confidence_score("TEST", "strat3")
    stake_t4 = engine.get_max_per_trade_pct("TEST", "strat3")
    logger.info(f"Conf: {conf_t4:.4f}, Stake: {stake_t4:.4f}")
    assert abs(conf_t4 - 0.53) < 0.0001
    assert abs(stake_t4 - DEFAULT_INITIAL_STAKE_PCT) < 0.0001


    # --- Sentiment Test Cases ---
    logger.info("\n--- Sentiment Test Cases ---")

    # Scenario: Profitable LONG, POSITIVE sentiment (Aligned)
    engine = _reset_confidence_memory_file()
    logger.info("--- Test S1: Profitable LONG, POSITIVE sentiment (Aligned) ---")
    # Initial: conf=0.5, stake=0.1. AI reports 0.8, trade profit 5%. Sent: positive, Dir: long.
    # Standard adj (from Test 2 like): learned_conf=0.56, stake_pct=0.115
    # Sentiment nudge (aligned):
    # learned_conf_nudge = SENTIMENT_CONF_IMPACT_LEARNED (0.02) * ai_reported_confidence (0.8) = 0.016
    # stake_nudge_val = SENTIMENT_CONF_IMPACT_STAKE (0.005)
    # expected_learned_conf = 0.56 + 0.016 = 0.576
    # expected_stake_pct = 0.115 + 0.005 = 0.120
    await engine.update_confidence("SENT_TEST", "long_profit_pos_sent",
                                   ai_reported_confidence=0.8, trade_result_pct=0.05,
                                   sentiment_data_status="present_and_not_empty",
                                   first_sentiment_observed="positive", trade_direction="long")
    conf_s1 = engine.get_confidence_score("SENT_TEST", "long_profit_pos_sent")
    stake_s1 = engine.get_max_per_trade_pct("SENT_TEST", "long_profit_pos_sent")
    logger.info(f"S1 - Conf: {conf_s1:.4f}, Stake: {stake_s1:.4f}")
    assert abs(conf_s1 - 0.576) < 0.0001
    assert abs(stake_s1 - 0.120) < 0.0001

    # Scenario: Unprofitable LONG, POSITIVE sentiment (Misaligned)
    engine = _reset_confidence_memory_file()
    logger.info("--- Test S2: Unprofitable LONG, POSITIVE sentiment (Misaligned) ---")
    # Initial: conf=0.5, stake=0.1. AI reports 0.7, trade loss -3%. Sent: positive, Dir: long.
    # Standard adj (like Test 3 but with AI conf 0.7):
    # loss_penalty = min(0.03 * 5, 1.0) = 0.15
    # confidence_decrease = (0.7 * 0.05) + (0.15 * 0.1) = 0.035 + 0.015 = 0.05
    # learned_conf_before_sent = 0.5 - 0.05 = 0.45
    # stake_decrease = 0.01 + (0.15 * 0.02) = 0.01 + 0.003 = 0.013
    # stake_pct_before_sent = 0.1 - 0.013 = 0.087
    # Sentiment nudge (misaligned):
    # learned_conf_nudge = SENTIMENT_CONF_IMPACT_LEARNED (0.02) * ai_reported_confidence (0.7) = 0.014
    # stake_nudge_val = SENTIMENT_CONF_IMPACT_STAKE (0.005)
    # expected_learned_conf = 0.45 - 0.014 = 0.436
    # expected_stake_pct = 0.087 - 0.005 = 0.082
    await engine.update_confidence("SENT_TEST", "long_loss_pos_sent",
                                   ai_reported_confidence=0.7, trade_result_pct=-0.03,
                                   sentiment_data_status="present_and_not_empty",
                                   first_sentiment_observed="positive", trade_direction="long")
    conf_s2 = engine.get_confidence_score("SENT_TEST", "long_loss_pos_sent")
    stake_s2 = engine.get_max_per_trade_pct("SENT_TEST", "long_loss_pos_sent")
    logger.info(f"S2 - Conf: {conf_s2:.4f}, Stake: {stake_s2:.4f}")
    assert abs(conf_s2 - 0.436) < 0.0001
    assert abs(stake_s2 - 0.082) < 0.0001

    # Scenario: Profitable SHORT, NEGATIVE sentiment (Aligned)
    engine = _reset_confidence_memory_file()
    logger.info("--- Test S3: Profitable SHORT, NEGATIVE sentiment (Aligned) ---")
    # Initial: conf=0.5, stake=0.1. AI reports 0.8, trade profit 5%. Sent: negative, Dir: short.
    # Standard adj (profit, same as S1): learned_conf=0.56, stake_pct=0.115
    # Sentiment nudge (aligned):
    # learned_conf_nudge = SENTIMENT_CONF_IMPACT_LEARNED (0.02) * ai_reported_confidence (0.8) = 0.016
    # stake_nudge_val = SENTIMENT_CONF_IMPACT_STAKE (0.005)
    # expected_learned_conf = 0.56 + 0.016 = 0.576
    # expected_stake_pct = 0.115 + 0.005 = 0.120
    await engine.update_confidence("SENT_TEST", "short_profit_neg_sent",
                                   ai_reported_confidence=0.8, trade_result_pct=0.05,
                                   sentiment_data_status="present_and_not_empty",
                                   first_sentiment_observed="negative", trade_direction="short")
    conf_s3 = engine.get_confidence_score("SENT_TEST", "short_profit_neg_sent")
    stake_s3 = engine.get_max_per_trade_pct("SENT_TEST", "short_profit_neg_sent")
    logger.info(f"S3 - Conf: {conf_s3:.4f}, Stake: {stake_s3:.4f}")
    assert abs(conf_s3 - 0.576) < 0.0001
    assert abs(stake_s3 - 0.120) < 0.0001

    # Scenario: Unprofitable SHORT, NEGATIVE sentiment (Misaligned)
    engine = _reset_confidence_memory_file()
    logger.info("--- Test S4: Unprofitable SHORT, NEGATIVE sentiment (Misaligned) ---")
    # Initial: conf=0.5, stake=0.1. AI reports 0.7, trade loss -3%. Sent: negative, Dir: short.
    # Standard adj (loss, same as S2): learned_conf_before_sent = 0.45, stake_pct_before_sent = 0.087
    # Sentiment nudge (misaligned):
    # learned_conf_nudge = SENTIMENT_CONF_IMPACT_LEARNED (0.02) * ai_reported_confidence (0.7) = 0.014
    # stake_nudge_val = SENTIMENT_CONF_IMPACT_STAKE (0.005)
    # expected_learned_conf = 0.45 - 0.014 = 0.436
    # expected_stake_pct = 0.087 - 0.005 = 0.082
    await engine.update_confidence("SENT_TEST", "short_loss_neg_sent",
                                   ai_reported_confidence=0.7, trade_result_pct=-0.03,
                                   sentiment_data_status="present_and_not_empty",
                                   first_sentiment_observed="negative", trade_direction="short")
    conf_s4 = engine.get_confidence_score("SENT_TEST", "short_loss_neg_sent")
    stake_s4 = engine.get_max_per_trade_pct("SENT_TEST", "short_loss_neg_sent")
    logger.info(f"S4 - Conf: {conf_s4:.4f}, Stake: {stake_s4:.4f}")
    assert abs(conf_s4 - 0.436) < 0.0001
    assert abs(stake_s4 - 0.082) < 0.0001

    # Scenario: Profitable LONG, NEUTRAL sentiment
    engine = _reset_confidence_memory_file()
    logger.info("--- Test S5: Profitable LONG, NEUTRAL sentiment ---")
    # Standard adj (profit, same as S1): learned_conf=0.56, stake_pct=0.115
    # Sentiment nudge: Neutral sentiment -> no nudge.
    # expected_learned_conf = 0.56
    # expected_stake_pct = 0.115
    await engine.update_confidence("SENT_TEST", "long_profit_neutral_sent",
                                   ai_reported_confidence=0.8, trade_result_pct=0.05,
                                   sentiment_data_status="present_and_not_empty",
                                   first_sentiment_observed="neutral", trade_direction="long")
    conf_s5 = engine.get_confidence_score("SENT_TEST", "long_profit_neutral_sent")
    stake_s5 = engine.get_max_per_trade_pct("SENT_TEST", "long_profit_neutral_sent")
    logger.info(f"S5 - Conf: {conf_s5:.4f}, Stake: {stake_s5:.4f}")
    assert abs(conf_s5 - 0.56) < 0.0001
    assert abs(stake_s5 - 0.115) < 0.0001

    # Scenario: Profitable LONG, ABSENT sentiment data
    engine = _reset_confidence_memory_file()
    logger.info("--- Test S6: Profitable LONG, ABSENT sentiment data ---")
    # Standard adj (profit, same as S1): learned_conf=0.56, stake_pct=0.115
    # Sentiment nudge: Absent data -> no nudge.
    # expected_learned_conf = 0.56
    # expected_stake_pct = 0.115
    await engine.update_confidence("SENT_TEST", "long_profit_absent_sent",
                                   ai_reported_confidence=0.8, trade_result_pct=0.05,
                                   sentiment_data_status="absent", # Data absent
                                   first_sentiment_observed=None, trade_direction="long")
    conf_s6 = engine.get_confidence_score("SENT_TEST", "long_profit_absent_sent")
    stake_s6 = engine.get_max_per_trade_pct("SENT_TEST", "long_profit_absent_sent")
    logger.info(f"S6 - Conf: {conf_s6:.4f}, Stake: {stake_s6:.4f}")
    assert abs(conf_s6 - 0.56) < 0.0001
    assert abs(stake_s6 - 0.115) < 0.0001

    # Scenario: Profitable SHORT, POSITIVE sentiment (Misaligned)
    engine = _reset_confidence_memory_file()
    logger.info("--- Test S7: Profitable SHORT, POSITIVE sentiment (Misaligned) ---")
    # Initial: conf=0.5, stake=0.1. AI reports 0.8, trade profit 5%. Sent: positive, Dir: short.
    # Standard adj (profit, same as S1): learned_conf=0.56, stake_pct=0.115
    # Sentiment nudge (misaligned):
    # learned_conf_nudge = SENTIMENT_CONF_IMPACT_LEARNED (0.02) * ai_reported_confidence (0.8) = 0.016
    # stake_nudge_val = SENTIMENT_CONF_IMPACT_STAKE (0.005)
    # expected_learned_conf = 0.56 - 0.016 = 0.544
    # expected_stake_pct = 0.115 - 0.005 = 0.110
    await engine.update_confidence("SENT_TEST", "short_profit_pos_sent",
                                   ai_reported_confidence=0.8, trade_result_pct=0.05,
                                   sentiment_data_status="present_and_not_empty",
                                   first_sentiment_observed="positive", trade_direction="short")
    conf_s7 = engine.get_confidence_score("SENT_TEST", "short_profit_pos_sent")
    stake_s7 = engine.get_max_per_trade_pct("SENT_TEST", "short_profit_pos_sent")
    logger.info(f"S7 - Conf: {conf_s7:.4f}, Stake: {stake_s7:.4f}")
    assert abs(conf_s7 - 0.544) < 0.0001
    assert abs(stake_s7 - 0.110) < 0.0001

    # Scenario: Unprofitable SHORT, POSITIVE sentiment (Aligned)
    engine = _reset_confidence_memory_file()
    logger.info("--- Test S8: Unprofitable SHORT, POSITIVE sentiment (Aligned) ---")
    # Initial: conf=0.5, stake=0.1. AI reports 0.7, trade loss -3%. Sent: positive, Dir: short.
    # Standard adj (loss, same as S2): learned_conf_before_sent = 0.45, stake_pct_before_sent = 0.087
    # Sentiment nudge (aligned):
    # learned_conf_nudge = SENTIMENT_CONF_IMPACT_LEARNED (0.02) * ai_reported_confidence (0.7) = 0.014
    # stake_nudge_val = SENTIMENT_CONF_IMPACT_STAKE (0.005)
    # expected_learned_conf = 0.45 + 0.014 = 0.464
    # expected_stake_pct = 0.087 + 0.005 = 0.092
    await engine.update_confidence("SENT_TEST", "short_loss_pos_sent",
                                   ai_reported_confidence=0.7, trade_result_pct=-0.03,
                                   sentiment_data_status="present_and_not_empty",
                                   first_sentiment_observed="positive", trade_direction="short")
    conf_s8 = engine.get_confidence_score("SENT_TEST", "short_loss_pos_sent")
    stake_s8 = engine.get_max_per_trade_pct("SENT_TEST", "short_loss_pos_sent")
    logger.info(f"S8 - Conf: {conf_s8:.4f}, Stake: {stake_s8:.4f}")
    assert abs(conf_s8 - 0.464) < 0.0001
    assert abs(stake_s8 - 0.092) < 0.0001


    logger.info("\nAll ConfidenceEngine tests passed.")
    # Clean up memory file after test
    if os.path.exists(CONFIDENCE_MEMORY_FILE):
        os.remove(CONFIDENCE_MEMORY_FILE)

if __name__ == '__main__':
    asyncio.run(run_test_confidence_engine())
